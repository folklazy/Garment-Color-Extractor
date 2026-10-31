"""Optional T4-pilot model. Requires an explicit, separately supplied checkpoint.

Torch is optional for the base package. This is not the historical M5 ensemble.
"""
from pathlib import Path
import hashlib
import numpy as np
from .features import extract_features

ALGORITHM='T4_MLP_62_64_32_12_v1'
CATEGORIES=(5,7,9,12)

def build_model():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(62,64),torch.nn.ReLU(),
        torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,12))

class LearnedColorPredictor:
    """Explicit artifact loader; softmax values are not calibrated confidence."""
    def __init__(self,checkpoint,device='cpu',expected_sha256=None):
        import torch
        path=Path(checkpoint)
        if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError('Checkpoint identity differs')
        a=torch.load(path,map_location='cpu',weights_only=True)
        if a['algorithm_id']!=ALGORITHM or a['class_order']!=list(range(1,13)):
            raise ValueError('Artifact model/class order mismatch')
        self.mean=np.array(a['mean'],np.float32);self.scale=np.array(a['scale'],np.float32)
        if self.mean.shape!=(62,) or self.scale.shape!=(62,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale<=0):
            raise ValueError('Invalid train-only normalization')
        self.device=device;self.model=build_model();self.model.load_state_dict(a['state_dict'],strict=True)
        self.model.to(device).float().eval();self.artifact=a

    def predict_pixels(self,pixels_rgb,native_category):
        """Caller supplies pixels after the frozen shared erosion step."""
        import torch
        if native_category not in CATEGORIES:raise ValueError('Native garment category must be 5,7,9,12')
        if len(pixels_rgb)==0:return dict(status='no_data',color_id=0,softmax=None,calibrated_probability=None,algorithm_id=ALGORITHM)
        f=extract_features(pixels_rgb)
        x=np.r_[f.feature_vector_58d,np.array([float(native_category==c) for c in CATEGORIES],np.float32)]
        with torch.inference_mode():prob=torch.softmax(self.model(torch.from_numpy((x-self.mean)/self.scale).to(self.device)[None]),dim=1)[0].cpu().numpy()
        if not np.isfinite(prob).all():raise ValueError('Nonfinite model output')
        return dict(status='ok',color_id=int(prob.argmax()+1),softmax=prob.tolist(),calibrated_probability=None,algorithm_id=ALGORITHM)
