"""Bounded CCIHP color-learning pilot. GPU is used for the color MLP only."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,importlib.metadata,io,json,os,shutil,subprocess,sys,time,traceback,zipfile
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import cv2
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from garment_color.features import extract_features
from garment_color.preprocessor import MaskPreprocessor
from garment_color.classifier import PrototypeRuleClassifier

CATS=(5,7,9,12)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def order(ids,split,seed):return sorted(ids,key=lambda i:hashlib.sha256(f'{seed}:{split}:{i}'.encode()).digest())
def png(z,name):
 with Image.open(io.BytesIO(z.read(name))) as im:im.load();return np.array(im)
def eligible(color,mask):
 n=int(mask.sum());known=color[mask];known=known[(known>=1)&(known<=12)]
 if n<100:return None,'REGION_LT100'
 if len(known)<.8*n:return None,'CHARACTERIZED_FRACTION_LT80'
 counts=np.bincount(known,minlength=13);label=int(np.argmax(counts[1:])+1)
 if counts[label]<.8*len(known):return None,'MODAL_FRACTION_LT80'
 return label,None

def metrics(gt,pred):
 if not len(gt):return dict(count=0,strict_accuracy=None,macroF1_fixed12=None,confusion=None)
 gt=np.asarray(gt);pred=np.asarray(pred)
 if gt.shape!=pred.shape or np.any((gt<1)|(gt>12)|(pred<1)|(pred>12)):raise ValueError('Color IDs must be 1..12')
 h=np.bincount(12*(gt-1)+(pred-1),minlength=144).reshape(12,12);tp=np.diag(h);support=h.sum(1);ps=h.sum(0)
 f1=np.divide(2*tp,support+ps,out=np.zeros(12,float),where=support+ps>0)
 return dict(count=len(gt),strict_accuracy=float(tp.sum()/h.sum()),macroF1_fixed12=float(f1.mean()),
    macroF1_GT_supported=float(f1[support>0].mean()),confusion=h.tolist(),
    per_class=[dict(color_id=i+1,GT_support=int(support[i]),pred_support=int(ps[i]),F1=float(f1[i]),
    precision=float(tp[i]/ps[i]) if ps[i] else None,recall=float(tp[i]/support[i]) if support[i] else None) for i in range(12)])

def prepare(workspace,out):
 cfg=read(R/'configs/T4_COLOR_PILOT_PROTOCOL.json');seed=cfg['sampling']['seed'];out.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),R/'configs/T4_COLOR_PILOT_PROTOCOL.json']+list((R/'src/garment_color').glob('*.py'))
 zpath=workspace/'datasets/CCIHP_icip.zip';z=zipfile.ZipFile(zpath);train=z.read('CCIHP_icip/dataset/Training/train_id.txt').decode().split();val=z.read('CCIHP_icip/dataset/Validation/val_id.txt').decode().split()
 selected={};seen={};excluded=[]
 for partition,split,n,ordered in [('train','Training',300,order(train,'Training',seed)),('calibration','Training',50,order(train,'Training',seed)),('validation','Validation',100,order(val,'Validation',seed))]:
  entries=[]
  for iid in ordered:
   if partition=='calibration' and iid in {r['image_id'] for r in selected['train']}:continue
   path=workspace/'datasets/CIHP'/split/'Images'/(iid+'.jpg');h=sha(path)
   if h in seen:excluded.append(dict(split=split,image_id=iid,reason='EXACT_RGB_DUPLICATE',first_partition=seen[h]));continue
   seen[h]=partition;entries.append(dict(image_id=iid,split=split,RGB_sha256=h))
   if len(entries)==n:break
  assert len(entries)==n
  selected[partition]=entries
 assert not ({r['image_id'] for r in selected['train']}&{r['image_id'] for r in selected['calibration']})
 protocol=dict(cfg,run_id=out.name,created_utc=utc(),dataset_archive=dict(path=str(zpath),size=zpath.stat().st_size,mtime_ns=zpath.stat().st_mtime_ns),
  CCIHP_README_sha256=hashlib.sha256(z.read('CCIHP_icip/README.md')).hexdigest(),memberships=selected,duplicate_exclusions=excluded,
  source_hashes={str(f.relative_to(R)):sha(f) for f in sources},workspace=str(workspace),out=str(out),
  launch_base_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip())
 atomic(out/'FROZEN_RUN.json',protocol);atomic(out/'STATE.json',dict(status='PREPARED',stage='FROZEN_MEMBERSHIPS',complete=0));z.close()
 return protocol

def run(p):
 out=Path(p['out']);workspace=Path(p['workspace']);start=time.monotonic();pre=MaskPreprocessor();base=PrototypeRuleClassifier()
 def check():
  if time.monotonic()-start>7200:raise RuntimeError('2H_TIME_GUARD')
  if shutil.disk_usage(out).free<50*2**30:raise RuntimeError('DISK_GUARD')
  for f,h in p['source_hashes'].items():
   if sha(R/f)!=h:raise RuntimeError('SOURCE_DRIFT '+f)
  archive=Path(p['dataset_archive']['path']);assert [archive.stat().st_size,archive.stat().st_mtime_ns]==[p['dataset_archive']['size'],p['dataset_archive']['mtime_ns']]
 state=dict(status='RUNNING',run_id=out.name,start_utc=utc(),pid=os.getpid(),stage='FEATURES',GPU_model_calls=0,model_training=True)
 atomic(out/'STATE.json',state);datasets={};allrows=[];excluded=[]
 with zipfile.ZipFile(p['dataset_archive']['path']) as z:
  for partition,members in p['memberships'].items():
   X=[];Y=[];baseline=[];rows=[]
   for index,m in enumerate(members):
    check();iid=m['image_id'];split=m['split'];image=workspace/'datasets/CIHP'/split/'Images'/(iid+'.jpg');assert sha(image)==m['RGB_sha256']
    rgb=np.array(Image.open(image).convert('RGB'));prefix=f'CCIHP_icip/dataset/{split}'
    gt={k:png(z,f'{prefix}/{k}/{iid}.png') for k in ['Category_ids','Human_ids','Color_ids']}
    assert all(a.shape==rgb.shape[:2] for a in gt.values())
    assert set(np.unique(gt['Color_ids']))<=set(range(13))
    hashes={k:hashlib.sha256(z.read(f'{prefix}/{k}/{iid}.png')).hexdigest() for k in gt}
    for pid in sorted(set(np.unique(gt['Human_ids']))-{0}):
     for cat in CATS:
      mask=(gt['Human_ids']==pid)&(gt['Category_ids']==cat)
      if not mask.any():continue
      label,reason=eligible(gt['Color_ids'],mask)
      if reason:excluded.append(dict(partition=partition,image_id=iid,person_id=int(pid),category=cat,reason=reason));continue
      pixels=pre.extract_clean_pixels(rgb,mask,True);f=extract_features(pixels)
      kind='upper' if cat in (5,7) else 'lower';pred=base.classify(f,kind).color_id
      x=np.r_[f.feature_vector_58d,np.array([float(cat==c) for c in CATS],np.float32)]
      assert x.shape==(62,) and np.isfinite(x).all()
      row=dict(partition=partition,image_id=iid,person_id=int(pid),category=cat,label=label,RGB_sha256=m['RGB_sha256'],GT_sha256=hashes,
       original_pixels=int(mask.sum()),analyzed_pixels=f.total_pixels,baseline_prediction=pred)
      rows.append(row);X.append(x);Y.append(label);baseline.append(pred)
    state.update(stage='FEATURES_'+partition,complete_images=index+1,partition_expected=len(members));atomic(out/'STATE.json',state)
    if index%50==0:print('FEATURES',partition,index+1,'/',len(members),'regions',len(rows),flush=True)
   assert rows
   datasets[partition]=(np.asarray(X,np.float32),np.asarray(Y,np.int64),np.asarray(baseline,np.int64),rows)
   allrows+=rows;np.savez_compressed(out/(partition+'_features.npz'),X=datasets[partition][0],y=datasets[partition][1])
 atomic(out/'REGIONS.json',dict(accepted=allrows,excluded=excluded))
 atomic(out/'DATA_COUNTS.json',dict(accepted={k:len(v[1]) for k,v in datasets.items()},excluded_reasons=dict(Counter((x['reason'] for x in excluded))),
    class_support={k:np.bincount(v[1],minlength=13)[1:].tolist() for k,v in datasets.items()}))
 check();import torch
 if not torch.cuda.is_available():raise RuntimeError('T4_CUDA_UNAVAILABLE; no CPU fallback')
 if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Expected Tesla T4')
 free,total=torch.cuda.mem_get_info()
 if free<2048*2**20:raise RuntimeError('VRAM_GUARD')
 torch.manual_seed(20261101);np.random.seed(20261101);torch.cuda.manual_seed_all(20261101);torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 trainX,trainY,_,_=datasets['train'];mean=trainX.mean(0);scale=trainX.std(0);scale[scale<1e-6]=1.
 tensor={k:(torch.from_numpy((v[0]-mean)/scale).cuda(),torch.from_numpy(v[1]-1).cuda()) for k,v in datasets.items() if k!='validation'}
 from garment_color.learned import build_model
 model=build_model().cuda().float();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 lossfn=torch.nn.CrossEntropyLoss();best=float('inf');best_epoch=None;history=[];torch.cuda.reset_peak_memory_stats()
 state.update(stage='T4_TRAINING',device=torch.cuda.get_device_name(0),train_regions=len(trainY));atomic(out/'STATE.json',state)
 for epoch in range(1,41):
  check();model.train();x,y=tensor['train'];perm=torch.randperm(len(y),device='cuda');loss_sum=0.
  for idx in perm.split(128):
   optimizer.zero_grad(set_to_none=True);logits=model(x[idx]);loss=lossfn(logits,y[idx]);loss.backward();optimizer.step();loss_sum+=float(loss.item())*len(idx)
  model.eval()
  with torch.inference_mode():cal_loss=float(lossfn(model(tensor['calibration'][0]),tensor['calibration'][1]).item())
  if not np.isfinite(cal_loss):raise RuntimeError('NONFINITE_TRAINING')
  history.append(dict(epoch=epoch,train_loss=loss_sum/len(y),calibration_loss=cal_loss))
  if cal_loss<best:
   best=cal_loss;best_epoch=epoch
   artifact=dict(algorithm_id='T4_MLP_62_64_32_12_v1',state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
    mean=mean.tolist(),scale=scale.tolist(),class_order=list(range(1,13)),feature_schema='prototype_rules_v2_descriptor58+nativecat5_7_9_12_onehot',
    protocol_sha256=sha(R/'configs/T4_COLOR_PILOT_PROTOCOL.json'),frozen_run_sha256=sha(out/'FROZEN_RUN.json'),best_epoch=epoch)
   torch.save(artifact,out/'learned_color.pt')
  state.update(epoch=epoch,best_epoch=best_epoch);atomic(out/'STATE.json',state)
  if epoch%10==0:print('T4_EPOCH',epoch,'calibration loss',cal_loss,flush=True)
 artifact=torch.load(out/'learned_color.pt',map_location='cpu',weights_only=True);model.load_state_dict(artifact['state_dict'],strict=True);model.eval()
 valX,valY,valBase,valRows=datasets['validation'];vx=torch.from_numpy((valX-mean)/scale).cuda()
 with torch.inference_mode():raw=model(vx);prob=torch.softmax(raw,dim=1).cpu().numpy();pred=prob.argmax(1)+1
 if not np.isfinite(prob).all():raise RuntimeError('NONFINITE_VALIDATION')
 summary=dict(status='COMPLETE',protocol_id=p['protocol_id'],run_id=out.name,selected_epoch=best_epoch,
   populations={k:dict(images=len(p['memberships'][k]),regions=len(v[1])) for k,v in datasets.items()},
   results={'prototype_rules_v2':metrics(valY,valBase),'T4_MLP_62_64_32_12_v1':metrics(valY,pred)},
   per_category={str(cat):{'prototype_rules_v2':metrics(valY[[r['category']==cat for r in valRows]],valBase[[r['category']==cat for r in valRows]]),
    'T4_MLP_62_64_32_12_v1':metrics(valY[[r['category']==cat for r in valRows]],pred[[r['category']==cat for r in valRows]])} for cat in CATS},
   GPU=dict(name=torch.cuda.get_device_name(0),allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20,
    precision='FP32/no AMP/TF32',torch=torch.__version__,runtime_CUDA=torch.version.cuda),
   artifact=dict(path='learned_color.pt',sha256=sha(out/'learned_color.pt'),bytes=(out/'learned_color.pt').stat().st_size),
   caveats=['Exploratory small oracle-mask color task, not segmentation end-to-end or deployment acceptance','CCIHP upper/lower garment definitions can contain dress parts; not the non-Dress common parser task','Validation never used for training/scaler/checkpoint/threshold tuning; calibration selects epoch','No CCTV accuracy claim or probability calibration; no latency ranking'])
 np.savez_compressed(out/'validation_predictions.npz',gt=valY,baseline=valBase,prediction=pred,softmax=prob)
 atomic(out/'TRAINING_HISTORY.json',history);atomic(out/'SUMMARY.json',summary)
 atomic(out/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,packages={n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','opencv-python-headless','pillow']},CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
 state.update(status='COMPLETE',stage='VALIDATION_EXPORTED',end_utc=utc(),selected_epoch=best_epoch,artifact_sha256=summary['artifact']['sha256']);atomic(out/'STATE.json',state)
 print('PILOT_COMPLETE',json.dumps(summary['populations']),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workspace',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--prepare-only',action='store_true');args=a.parse_args()
 if args.output.exists():p=read(args.output/'FROZEN_RUN.json')
 else:p=prepare(args.workspace.resolve(),args.output.resolve())
 if not args.prepare_only:
  lock=(Path(p['out'])/'RUN.lock').open('a')
  import fcntl;fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (Path(p['out'])/'EXECUTION_STARTED.json').exists():raise RuntimeError('Attempt already started; reconcile rather than retry')
  atomic(Path(p['out'])/'EXECUTION_STARTED.json',dict(utc=utc(),pid=os.getpid(),launch_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()))
  try:run(p)
  except BaseException:
   atomic(Path(p['out'])/'FAILURE.json',dict(status='FAILED',error=traceback.format_exc(),utc=utc()));raise
