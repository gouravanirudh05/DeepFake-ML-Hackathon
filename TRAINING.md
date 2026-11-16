# Training & inference guide

Approach: frozen foundation ViT-L backbones (**CLIP**, **DINOv3**, DINOv2, Perception Encoder) with
parameter-efficient tuning, either **LoRA** or **Effort-style SVD subspace tuning** (ICML 2025). The detector
reads multi-layer CLS features (RINE, ECCV 2024) and is trained only on the provided data. Training adds
**self-blended pseudo-fakes** (SBI, CVPR 2022), a small-loss label-noise filter, robustness augmentations
and leakage-free grouped folds. Adapters are merged into the weights after training, so inference runs a
plain timm ViT. The final model is an ensemble of diverse backbones.

| component | what it does | why |
|---|---|---|
| `--feat_layers auto` | CLS token after blocks 3,7,…,23, each LayerNorm-ed and concatenated | mid layers keep the low-level blending and texture cues that the last layer drops |
| `--adapter svd` (`src/svd_adapter.py`) | SVD of every attention weight; the top singular directions stay frozen and only the smallest `--svd_rank` directions are trained, with orthogonality and norm-preservation losses | keeps pretrained knowledge intact, so the model generalises to unseen forgery methods (Effort) |
| `--sbi_prob` (`src/sbi.py`) | a real image is blended with a colour, sharpness or position-jittered copy of itself inside a deformed face mask, then labelled fake | teaches method-agnostic swap traces using only the provided real images |
| `--drop_rate` | ignores the highest-loss fraction of each batch | likely mislabelled images (~8% confident errors) stop dominating the gradient |

## 0. Setup (Python 3.10)
```sh
pip install -r requirements.txt
```

## 1. Data audit + folds (already committed as `splits/folds.csv`)
```sh
python audit_and_split.py --data_dir Data --out splits/folds.csv
```
pHash near-duplicate grouping, which found 295 near-dup pairs and 53 real/fake pairs of the same source photo.
`StratifiedGroupKFold` keeps every group inside one fold.

## Kaggle workflow (CLI)
- Data: private dataset `gouravanirudh/deepfake-hackathon-data` (also `gouravanirudh1105/...`, the account used for GPU runs; token `~/.kaggle/access_token_gpu`, pass via `KAGGLE_API_TOKEN`) (train/, test/, metadata, folds.csv).
- Notebooks are generated from the repo sources by experiment preset (see `PRESETS` in
  `kaggle/make_notebook.py`); each preset becomes its own kernel in `kaggle/kernels/<slug>/`, so two
  presets can run in parallel:
```sh
python kaggle/make_notebook.py --preset sota-clip [--folds 0,1,2,3,4]   # re-embed sources after any change
kaggle kernels push -p kaggle/kernels/deepfake-sota-clip --accelerator NvidiaTeslaT4
kaggle kernels status gouravanirudh1105/deepfake-sota-clip
kaggle kernels output gouravanirudh1105/deepfake-sota-clip -p kaggle/output_deepfake-sota-clip
```

## 2. Train (Kaggle / Colab, T4 16 GB; internet ON to fetch backbone weights)
```sh
# one fold, ~30-40 min on a T4 (estimate). If you hit OOM, use --bs 16 --accum 2.
python train.py --data_dir /kaggle/input/<dataset>/Data --fold 0 \
    --arch vit_large_patch14_clip_224.openai --bs 32 --grad_ckpt --epochs 10
```
Outputs land in `outputs/<run_name>/fold<k>/`: `history.csv` (curves), `robustness.json`
(AUC per degradation), `oof.csv`, and `member.pt` (merged fp16 weights).

Things to try (log every run for the report's ablation table):
| knob | values |
|---|---|
| `--arch` | `vit_large_patch14_clip_224.openai`, `vit_large_patch16_dinov3.lvd1689m --img_size 224`, `vit_large_patch14_reg4_dinov2.lvd142m --img_size 224`, `vit_pe_core_large_patch14_336.fb --img_size 224` |
| `--adapter` | `lora` (`--r`, `--alpha`), `svd` (`--svd_rank 4 --targets attn --lr 2e-4`, `--svd_reg 1.0`) |
| `--sbi_prob` | 0, 0.25 |
| `--r` / `--targets` | 8 / 16 / 32; `attn` vs `all` |
| `--aug_strength` | 0, 0.5, 1.0, 1.5 |
| `--label_smoothing` | 0, 0.05, 0.1 |
| `--feat_layers` | `''` (final CLS), `auto` (CLS of blocks 3,7,…,23, each LayerNorm-ed and concatenated), `7,15,23` |
| `--loss` | `bce`, `gce` (Generalized Cross-Entropy, `--gce_q 0.7`, tolerant to label noise) |
| `--drop_rate` | 0, 0.05, 0.08: drop that fraction of highest-loss samples per batch (ramped over `--drop_warmup` epochs) |

### Experiment log (fold 0, CLIP ViT-L/14 openai, LoRA r=16 all)
| run | changes | clean AUC | mean degraded AUC |
|---|---|---|---|
| baseline | final CLS, BCE+LS 0.05, 10 ep (best ep 4) | 0.9134 | 0.9122 |
| ml5 | multi-layer CLS (`auto`), 5 ep | 0.9140 | 0.9123 |
| ml5_drop | ml5 + `--drop_rate 0.08` | **0.9166** | **0.9152** |
| ml5_gce | ml5 + `--loss gce` | 0.9109 | 0.9118 |
| ensemble | mean logit of baseline + ml5_drop (OOF) | **0.9201** | n/a |

Paired bootstrap on the same 1,412 images: every single-model difference vs baseline is inside
±0.01 (95% CI), so none is significant on one fold. The baseline+ml5_drop ensemble gain comes from
diversity: those two members have the lowest logit correlation (0.906).

## 3. Bundle + inference
```sh
python build_bundle.py --members "outputs/clip_l14_lora/fold*/member.pt" \
    --oof "outputs/clip_l14_lora/fold*/oof.csv" --out model_weights.pt
python inference_script.py --test_dir Data/test/images --weights model_weights.pt --output predictions.csv
```
The threshold is chosen by Youden's J on out-of-fold predictions. Time the inference script on a T4 before
submitting. The efficiency score needs < 5 min, and each ViT-L member costs roughly 30-60 s for 3k images
with flip TTA (estimate; measure it).
