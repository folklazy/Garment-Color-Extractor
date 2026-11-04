# Data used by the observed-color study

Source: A. Loesch and R. Audigier, **Describe Me If You Can! Characterized Instance-Level Human Parsing**, ICIP2021, pp2528–2532, [DOI](https://doi.org/10.1109/ICIP42928.2021.9506509). The locally supplied `CCIHP_icip.zip` README was read directly; its SHA256 is `4b73782345d94752e3a4a015ca0f8c9aa6bffe38d9568af7018071160f00bc5f`. This is an observed local source identity, not a publisher signature.

CCIHP uses the same RGB image collection as [CIHP / LIP](https://www.sysuhcp.com/lip/index.php); the archive README points to the [official image collection](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M?usp=sharing). No download or redistribution was performed for this work unit. Shared local locations are `Human-Body-Extraction/datasets/CCIHP_icip.zip` and `datasets/CIHP/{Training,Validation}/Images/`. Local paths are asset bindings, not public hosting.

The annotation release contains Category_ids, Human_ids and Color_ids maps. Our task selects a modal **color label for an oracle garment region**; it does not use the official characterized parsing AP evaluator. Category5 UpperClothes and12Skirt may represent top/bottom of dresses; CCIHP taxonomy is not identical to CIHP20 or the non-Dress common-view parsing task. Native categories5/7/9/12 and colors1–12 are retained. Color0 is uncharacterized, rather than an observed gray/background color. Human0 is outside a labeled person. No semantic-pixel metric or model-specific ignore domain is invented here.

Exact membership and exclusions are inherited from [accepted pilot provenance](reports/t4-pilot-20261101/MANIFEST.json). Current [protocol](EXPERIMENT_PROTOCOL.md) is the single authority for region eligibility, partition use, erosion and scoring. Source GT is read-only; only region-level modal references/derived summaries are produced. This study is a subset of100 official Validation images, not the full5,000. No claim of complete upstream training/pretraining-exposure control is made.

## Attribution and distribution

The archive states **annotations: CC BY-NC4.0; evaluation code: MIT**. Dataset-derived reference fields such as CCIHP IDs/modal GT colors in published per-region prediction tables remain subject to the annotation terms and require attribution to Loesch/Audigier. Modifications: selecting the frozen subset and deriving modal color labels per eligible person×native garment region; original pixel annotations/images are not distributed. See [CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The repository's existing MIT license for first-party code is unchanged; it does not relicense dataset annotations, images, trained weights or other external artifacts. Models trained here are stored locally and not automatically licensed/distributed. No claim of commercial authorization is made by placing derived study tables beside MIT code.

CCTV source images/annotations/saved parsing outputs are separate restricted assets with mixed or unknown color-label provenance. They are absent from this comparison's training/evaluation and from public Git. The earlier private pilot gallery is qualitative-only and has no independent color-accuracy result.
