# Thermal Image Processing and Computer Vision for Anomaly Detection and Power Loss Estimation in Photovoltaic Arrays

**ระบบประมวลผลภาพความร้อนเพื่อตรวจจับความผิดปกติและประเมินการสูญเสียกำลังไฟฟ้าของแผงเซลล์สุริยะ**

Senior project 2102491, Department of Electrical Engineering, Faculty of Engineering, Chulalongkorn University (academic year 2569 / 2026)
Student: นายติณณ์ ลวกุล · Advisor: ผศ.ดร.สุรีย์ พุ่มรินทร์

## Overview

The system takes infrared (IR) drone images of a PV array and estimates how much power is lost:

1. **Detect panels** in the full-frame image (YOLO, next semester)
2. **Prepare** each panel image (native 40×24, per-image z-score, no resizing)
3. **Classify** the anomaly into 12 classes with **ResNet18S**
4. **Estimate power loss** from the number of bypassed substrings *k*: `P_loss = (k / 3) · P_rated`

| Class | k |
|---|---|
| No-Anomaly, cell-level faults (Cell, Cell-Multi, Cracking, Hot-Spot, Hot-Spot-Multi), external (Shadowing, Vegetation, Soiling) | 0 |
| Diode | 1 |
| Diode-Multi | 2 |
| Offline-Module | 3 |

The power-loss equation was checked against measured I-V curves of the DTU dataset (Notebook 04).

## Final model (this semester)

ResNet18S (ResNet-18 adapted to 40×24 input: 3×3 stride-1 `conv1`, no max-pool, ImageNet initialisation) trained with **mixup** (α = 0.2) for 30 epochs, ensemble of 3 seeds + test-time augmentation (4 flips). Runs R131–R133.

Test set (3,000 images, used once):

| Metric | Single model (mean of 3 seeds) | Ensemble + TTA |
|---|---|---|
| Accuracy (12 classes) | 0.835 | **0.851** |
| Macro-F1 (12 classes) | 0.713 | **0.738** (95% CI 0.709–0.763) |
| Macro-F1 (5 groups) | 0.840 | 0.851 |
| Fault / no-fault accuracy | 0.947 | 0.954 |
| Accuracy of k | 0.968 | 0.970 |
| Error of total power loss | +6.9% | +3.5% |
| Test loss (NLL) | 0.545 | 0.477 |

Model selection used repeated paired stratified 5-fold cross-validation (6 repeats, 30 folds) with the corrected resampled t-test (Nadeau & Bengio, 2003).

## Repository layout

```
├── 01_baseline_classical.ipynb … 12_mixup_60_epochs.ipynb   notebooks (with outputs)
├── splits.csv              fixed 70/15/15 train/val/test split (seed 42)
├── experiment_log.xlsx     every run (R01, R02, …) with all its settings and results
├── results/                result tables (.csv) and decisions (.txt) written by the notebooks
├── figures/                figures written by the notebooks (loss curves of every run, confusion matrices, …)
└── requirements.txt
```

| Notebook | What it does | Runs |
|---|---|---|
| 01 | Classical baseline: hand-crafted features + classical ML | – |
| 02 | First CNN in PyTorch | – |
| 03 | Short CNN demo | – |
| 04 | DTU data: ΔT vs. measured power loss, checks `P_loss = (k/3)·P_rated` | – |
| 05 | Class balancing and transfer learning (ResNet-18/50, EfficientNet-B0); `05_balanced_resnet_v2` is the complete run | R05–R25 |
| 06 | Native 40×24 input with architectures for small images (ResNet18S, WRN-16-4, ResNet-20, DenseNet-BC, …), with and without regularization | R26–R61 |
| 07 | Full metrics (mAP, MCC, ECE, McNemar, …) for every saved model, no retraining | – |
| 08 | Early stopping, ResNet10S, regularization without label smoothing | R62–R70 |
| 09 | ResNet18S vs. ResNet10S by repeated paired cross-validation | R71–R130 |
| 10 | mixup, CutMix, knowledge distillation, temperature scaling, label check (confident learning) | R131–R139 |
| 11 | ResNet18S base vs. + mixup by repeated paired cross-validation | R140–R169 |
| 12 | mixup with 60 instead of 30 epochs | R170–R172 |

## How to run

1. Install the packages in `requirements.txt` (install PyTorch for your CUDA version first).
2. Download the Raptor Maps dataset and place it so that one of these paths exists, relative to the notebooks:
   `archive/2020-02-14_InfraredSolarModules/InfraredSolarModules/images` (also `../archive/…` or `../data/archive/…`).
3. Close `experiment_log.xlsx`, then *Run All*. Every run gets a new ID and never overwrites an earlier one.
   Re-running writes result files next to the notebooks, plus `checkpoints/`, `runs/` and `test_probs/` (not tracked by git).

The notebooks print the device at start-up (`cuda` and the GPU name) and save it in each run's `config.json`.

## Datasets

| Dataset | Use | Source |
|---|---|---|
| Raptor Maps InfraredSolarModules (20,000 images, 40×24, 12 classes) | train and test the classifier | https://github.com/RaptorMaps/InfraredSolarModules |
| DTU PV plant – Drone Infrared Thermography Dataset (R. Del Prado Santamaria, 2025) | check the power-loss equation with measured I-V curves | https://doi.org/10.11583/DTU.29665346 |
| Thermal PV Panel Detection Dataset for UAV-Based Inspection (Christakakis, Pechlivani, Dimou, 2025, CC BY 4.0) | train YOLO for panel detection (next semester) | https://doi.org/10.5281/zenodo.16420123 |

The images are not included in this repository; please download them from the sources above.

## Use of AI tools

This project used Claude (Anthropic) as an assistant for:

- **Code:** writing and revising the notebook code (data loading, models, training loops, evaluation and statistics) from the experiments I specified.
- **Literature:** finding and summarising papers on overfitting, model evaluation and PV fault detection. Every reference cited in the report was checked against its original source.
- **Writing:** drafting and formatting the project report and this README.

My own work:

- Defining the project scope and the follow-up experiments with my advisor, based on the advisor's feedback at each meeting.
- Running every experiment (R01–R172) on my own GPU and checking the outputs.
- Interpreting the results and choosing the final model. The decision rules for model selection were fixed before the cross-validation runs (Notebooks 09 and 11).
- Reviewing and editing the code and the report. I am responsible for all results and any errors in this repository.
