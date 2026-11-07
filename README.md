# ChestXray12 Classification

This repository contains code for 12-class chest X-ray classification using different lung preprocessing methods, lung segmentation models, and explainability methods.

## Experiments
Each experiment is managed via YAML file. 
- experimentA.yaml
  Balanced data, random crop, 256 x 256 → 224 x 224, Categorical Crossentropy
- experimentB.yaml
  All data used, random crop, 256 x 256 → 224 x 224, Categorical Crossentropy
- experimentC.yaml
  All data used, no random crop, 256 x 256 → 224 x 224, weighted Categorical Crossentropy
- experimentD.yaml
  All data used, random crop, 384 x 384 → 384 x 384, weighted Categorical Crossentropy
- experimentE.yaml
  All data used, no random crop, 384 x 384 → 384 x 384, weighted Categorical Crossentropy
- experimentF.yaml
  Balanced data, no random crop, 256 x 256 → 224 x 224, Categorical Crossentropy
- experimentG.yaml
  3 class classification (Fibrosis, Pneumonia, Inflitration)

## Dataset

The project uses a 12-class chest X-ray dataset with separate `train` and `test` directories.

Expected dataset structure:

```text
ChestXray12/
└── images/
    ├── train/
    │   ├── Atelectasis/
    │   ├── Cardiomegaly/
    │   ├── Consolidation/
    │   ├── Edema/
    │   ├── Effusion/
    │   ├── Emphysema/
    │   ├── Fibrosis/
    │   ├── Infiltration/
    │   ├── Mass/
    │   ├── Nodule/
    │   ├── Pneumonia/
    │   └── Pneumothorax/
    │
    └── test/
        ├── Atelectasis/
        ├── Cardiomegaly/
        ├── Consolidation/
        ├── Edema/
        ├── Effusion/
        ├── Emphysema/
        ├── Fibrosis/
        ├── Infiltration/
        ├── Mass/
        ├── Nodule/
        ├── Pneumonia/
        └── Pneumothorax/
```

The 12 classes are:

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Effusion
Emphysema
Fibrosis
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
```

The dataset root is specified in:

```text
configs/classification.yaml
```

Example:

```yaml
dataset:
  root: /media/share/Member/ueki/datasets/ChestXray12/images
```

### Balanced dataset

The experiments use class-balanced data.

`Pneumonia` is used as the reference class, and the other classes are randomly downsampled to the same number of images.

Balancing is performed separately for the original `train` and `test` sets.

The balanced training set is further divided into training and validation subsets. The test set is kept for final evaluation.

A separate manifest is created for each random seed, for example:

```text
outputs/manifests/
├── seed42.csv
├── seed43.csv
├── seed44.csv
├── seed45.csv
└── seed46.csv
```

The same manifest is used across preprocessing conditions so that the same images are compared between experiments.

---

## How to Run

Run the following commands from the repository root:

```bash
cd /media/share/Member/ueki/master/chestxray12_classification
```

### 1. Generate lung masks

```bash
python scripts/generate_masks.py \
    --config configs/masks.yaml
```

This script generates lung segmentation masks for the chest X-ray images.

Supported segmentation models include:

```text
xrv_pspnet
xrv_cxas
niaid_resunet
```

The generated masks are stored under the mask root configured in `configs/masks.yaml`.

Example structure:

```text
masks/
├── xrv_pspnet/
│   └── whole_lung/
│
├── xrv_cxas/
│   ├── left_lung/
│   ├── right_lung/
│   └── whole_lung/
│
└── niaid_resunet/
    └── whole_lung/
```

The original train/test/class directory structure is preserved under each mask directory.

---

### 2. Create balanced manifests

```bash
python scripts/create_balanced_manifest.py \
    --config configs/classification.yaml
```

This script:

- reads the original `train` and `test` datasets,
- balances all classes to the configured reference class,
- creates the train/validation split,
- keeps the test set for final evaluation,
- creates one manifest for each configured random seed.

The generated manifests are saved under:

```text
outputs/manifests/
```

Run this script before classifier training.

---

### 3. Train and evaluate classifiers

```bash
python scripts/train_classifier.py \
    --config configs/classification.yaml
```

This script trains and evaluates all enabled classification experiments defined in `configs/classification.yaml`.

Supported classifier backbones include:

```text
ResNet50
DenseNet201
ViT-B/16
```

The preprocessing experiments can include:

```text
original
pspnet_lung_only
cxas_lung_only
niaid_lung_only
pspnet_moment
cxas_moment
niaid_moment
```

The script runs the enabled backbones, experiments, and random seeds specified in the configuration file.

Classification outputs are stored under:

```text
outputs/classification/
```

Example:

```text
outputs/classification/
└── densenet201/
    ├── original/
    │   └── seed42/
    ├── pspnet_lung_only/
    │   └── seed42/
    ├── cxas_lung_only/
    │   └── seed42/
    └── ...
```

Each run can contain files such as:

```text
checkpoints/best_model.pth
test_metrics.json
test_predictions.csv
confusion_matrix.png
```

---

### 4. Generate XAI visualizations

```bash
python scripts/run_xai.py \
    --config configs/xai.yaml
```

This script applies explainability methods to trained classifiers.

The XAI target is the model's predicted class.

Supported methods are:

```text
CNN models:
- Grad-CAM
- Grad-CAM++
- LIME
- SHAP

ViT:
- LIME
- SHAP
- Attention Rollout
```

The same selected test images are used across preprocessing experiments for easier comparison.

The generated XAI images are stored under:

```text
outputs/xai/
```

Example:

```text
outputs/xai/
└── densenet201/
    └── original/
        └── seed42/
            ├── *_xai.png
            └── xai_metadata.csv
```

Each XAI figure contains the input image, visualization results, ground-truth class, predicted class, and predicted probability.

---

### 5. Compare XAI results across preprocessing experiments

```bash
python scripts/compare_xai.py \
    --config configs/xai.yaml
```

This script compares XAI results for the same image across different preprocessing experiments.

It uses previously generated XAI outputs and does not retrain the classifiers.

This is useful for comparing conditions such as:

```text
Original
PSPNet lung-only
CXAS lung-only
NIAID lung-only
Moment-standardized conditions
```

---

### 6. Create PDF reports from XAI images

```bash
python scripts/create_xai_pdf.py \
    --config configs/xai_pdf.yaml
```

This script collects the generated XAI images and creates a PDF for each:

```text
backbone x experiment x seed
```

For example:

```text
outputs/xai/
└── densenet201/
    └── original/
        ├── seed42/
        │   └── *_xai.png
        ├── seed43/
        │   └── *_xai.png
        ├── seed42.pdf
        └── seed43.pdf
```

The PDF is saved directly under each experiment directory.

The current PDF layout places four XAI result images vertically on each page.

---

## Typical Workflow

```bash
# 1. Generate lung masks
python scripts/generate_masks.py \
    --config configs/masks.yaml

# 2. Create balanced manifests
python scripts/create_balanced_manifest.py \
    --config configs/classification.yaml

# 3. Train and evaluate classifiers
python scripts/train_classifier.py \
    --config configs/classification/experiment{A-G}.yaml

# 4. Generate XAI results
python scripts/run_xai.py \
    --config configs/xai.yaml

# 5. Compare XAI results
python scripts/compare_xai.py \
    --config configs/xai.yaml

# 6. Create XAI PDF reports
python scripts/create_xai_pdf.py \
    --config configs/xai_pdf.yaml
```
