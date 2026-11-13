# Terrain Latent Classifier

## Purpose

Add a reusable script module for predicting a terrain label from the concatenated latent representation:

```python
torch.cat([long_latent, short_latent], dim=-1)
```

The module uses the same recorded latent data and terrain labeling rule as the latent UMAP/t-SNE analysis script. It is intended for quick supervised probing of whether the learned latent representation separates `plane`, `stair`, and `slope`.

## File Added

- `scripts/terrain_latent_classifier.py`

## Functionality

The script supports two command-line modes:

- `train`: load `latents.pt`, derive terrain labels from positions, train an MLP classifier, and save a checkpoint.
- `predict`: load a saved classifier checkpoint and run inference on a `latents.pt` file.

It also exposes a Python API:

- `TerrainLatentClassifier.load(...)`
- `TerrainLatentClassifier.predict(...)`
- `TerrainLatentClassifier.predict_proba(...)`
- `TerrainLatentClassifier.predict_concat(...)`
- `TerrainLatentClassifier.predict_proba_concat(...)`

## Terrain Label Rule

Labels are derived from recorded `positions` using the same hand-written rule as the visualization script:

```text
x < 4.0 or x > 6.0              -> plane
4.0 <= x <= 6.0 and y < 4.0     -> stair
4.0 <= x <= 6.0 and y >= 4.0    -> slope
```

Label ids:

```text
0 -> plane
1 -> stair
2 -> slope
```

## Data Layout

Expected `latents.pt` keys:

```python
long_latents
short_latents
positions
```

Observed data shape for the initial target file:

```text
long_latents:  (9851, 1, 32)
short_latents: (9851, 1, 32)
positions:     (9851, 2)
```

After squeezing and concatenation, the classifier input dimension is `64`.

The script accepts either the direct file path:

```text
logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50/state_recorder/latents.pt
```

or the run directory:

```text
logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50
```

When a directory is passed, the script resolves:

```text
<run_dir>/state_recorder/latents.pt
```

## Training Instruction

Basic training:

```bash
python scripts/terrain_latent_classifier.py train \
  --latents-path logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50 \
  --save-path logs/terrain_latent_classifier.pt \
  --epochs 200 \
  --batch-size 256
```

Useful options:

```text
--hidden-dims 128,128
--dropout 0.1
--lr 1e-3
--weight-decay 1e-4
--val-ratio 0.2
--device cpu
--device cuda
```

The checkpoint stores:

- model weights
- input dimension
- hidden dimensions
- dropout
- feature normalization mean/std
- label mappings
- terrain label rule description
- validation confusion matrix

## Prediction Instruction

Run prediction on a latent recording:

```bash
python scripts/terrain_latent_classifier.py predict \
  --checkpoint logs/terrain_latent_classifier.pt \
  --latents-path logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50
```

Optionally save predictions:

```bash
python scripts/terrain_latent_classifier.py predict \
  --checkpoint logs/terrain_latent_classifier.pt \
  --latents-path logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50 \
  --output-path logs/terrain_latent_predictions.pt
```

Saved prediction output contains:

```python
pred_ids
pred_labels
probs
true_ids_from_positions
```

## Python Usage

Load the classifier and infer from separate long/short latents:

```python
import torch
from scripts.terrain_latent_classifier import TerrainLatentClassifier

classifier = TerrainLatentClassifier.load(
    "logs/terrain_latent_classifier.pt",
    device="cpu",
)

latents = torch.load(
    "logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50/state_recorder/latents.pt",
    map_location="cpu",
)

labels = classifier.predict(
    latents["long_latents"],
    latents["short_latents"],
)
probs = classifier.predict_proba(
    latents["long_latents"],
    latents["short_latents"],
)
```

Infer from an already concatenated latent tensor:

```python
long_latents = latents["long_latents"].squeeze(1)
short_latents = latents["short_latents"].squeeze(1)
concat_latents = torch.cat([long_latents, short_latents], dim=-1)

labels = classifier.predict_concat(concat_latents)
probs = classifier.predict_proba_concat(concat_latents)
```

Single-sample inference:

```python
label = classifier.predict_concat(concat_latents[0])[0]
prob = classifier.predict_proba_concat(concat_latents[0])[0]
```

`prob` follows label id order:

```text
0 -> plane
1 -> stair
2 -> slope
```

## Verification

Syntax and CLI checks passed:

```bash
python -m py_compile scripts/terrain_latent_classifier.py
python scripts/terrain_latent_classifier.py --help
```

Smoke training on the real latent file passed:

```bash
python scripts/terrain_latent_classifier.py train \
  --epochs 2 \
  --batch-size 512 \
  --device cpu \
  --save-path /tmp/terrain_latent_classifier_smoke.pt \
  --log-every 1
```

Observed smoke result:

```text
Loaded 9851 samples
Input dim: 64
Terrain counts: {"plane": 6573, "slope": 1749, "stair": 1529}
Train/val split: 7880/1971
epoch 0002/2 loss=0.506653 train_acc=0.8315 val_acc=0.8148
```

Checkpoint reload and prediction passed:

```bash
python scripts/terrain_latent_classifier.py predict \
  --checkpoint /tmp/terrain_latent_classifier_smoke.pt \
  --device cpu
```

Observed prediction result:

```text
Predicted 9851 samples
Accuracy against position-derived labels: 0.8281
Prediction counts: {"plane": 6611, "slope": 1159, "stair": 2081}
```

Directory-style `--latents-path` was also verified:

```bash
python scripts/terrain_latent_classifier.py train \
  --latents-path logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50 \
  --epochs 1 \
  --batch-size 512 \
  --device cpu \
  --save-path /tmp/terrain_latent_classifier_dir_smoke.pt \
  --log-every 1
```
