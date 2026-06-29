# PhysioFuse

Official code-style release package for **PhysioFuse: Physiological Consistency Guided Multi-Site Visual Learning for Non-Invasive Hemoglobin Estimation**.

This repository is intentionally minimal. It contains source code and configuration files for the paper model. The full clinical dataset, full ROI mask set, sample-level labels, predictions, and trained checkpoints are not publicly released because they are subject to institutional data-use and ethics restrictions. A tiny anonymized `example_data/` folder is provided only to illustrate the expected file layout.

## Data Format

The paper used a private clinical dataset of **995 subjects** from Southwest Hospital. Each subject has a paired conjunctival/eyelid ROI image, a paired nailfold ROI image, optional binary ROI masks for both anatomical sites, and a ground-truth hemoglobin value from concurrent complete blood count testing.

The code expects data to be placed locally as:

```text
data/data-new/
  hbs.csv
  conjunctiva/
    <sample_id>.jpg
    conjunctiva_masks/
      <sample_id>.png
  nail/
    <sample_id>.jpg
    nail_masks/
      <sample_id>.png
```

`hbs.csv` must contain:

```text
name,hb
```

Mask files are optional for code execution. If a mask file is missing, the loader uses a full-image mask. The paper experiments used segmented/masked ROI inputs resized to `224 x 224`.

Do not commit the full `data/` directory, full clinical labels, the full ROI mask set, predictions, checkpoints, or sample-level outputs.

## Layout

```text
PhysioFuse/
  README.md
  LICENSE
  requirements.txt
  configs/
    physiofuse.yaml
  src/
    physiofuse/
      __init__.py
      model.py
      dataset.py
      losses.py
      train.py
      evaluate.py
  scripts/
    train_cv.py
    evaluate_cv.py
    infer.py
  example_data/
    README.md
    hbs.csv
    conjunctiva/
      sample_001.jpg
      sample_002.jpg
      conjunctiva_masks/
        sample_001.png
        sample_002.png
    nail/
      sample_001.jpg
      sample_002.jpg
      nail_masks/
        sample_001.png
        sample_002.png
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If needed, install the PyTorch build matching your CUDA version before installing the remaining requirements.

## Train

```bash
python scripts/train_cv.py --config configs/physiofuse.yaml
```

The default config runs 5-fold stratified cross-validation and writes checkpoints and fold summaries under `outputs/physiofuse_cv/`.

## Evaluate Checkpoints

Trained checkpoints are not included in this repository. If you have approved access to checkpoints, place them locally and run:

```bash
python scripts/evaluate_cv.py --config configs/physiofuse.yaml --checkpoint-dir checkpoints
```

Expected checkpoint names:

```text
physiofuse_best_fold0.pth
physiofuse_best_fold1.pth
...
```

## Single Inference

```bash
python scripts/infer.py \
  --config configs/physiofuse.yaml \
  --checkpoint checkpoints/physiofuse_best_fold0.pth \
  --eye path/to/conjunctiva.jpg \
  --nail path/to/nail.jpg
```

Optional mask arguments are also supported:

```bash
--eye-mask path/to/conjunctiva_mask.png --nail-mask path/to/nail_mask.png
```

## Results

Aggregate numerical results are reported in the manuscript. This repository does not include subject-level labels, prediction arrays, local paths, or other sample-level clinical outputs.
