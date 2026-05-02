# Face Mask CV From Scratch

Computer vision pipeline for binary face mask classification using classical feature extraction and a fully custom CNN trained from scratch.

This project was built for an academic computer vision practicum. It intentionally avoids pretrained models and transfer learning: no ResNet, MobileNet, EfficientNet, VGG pretrained variants, `tf.keras.applications`, or ImageNet weights are used.

## Highlights

- Dataset: Kaggle Face Mask Dataset (`omkargurav/face-mask-dataset`)
- Classes: `with_mask`, `without_mask`
- Split: train/validation/test = `70/15/15`
- Input size: `128x128`
- Enhancement/restoration: CLAHE + light Gaussian denoising
- Face crop: OpenCV Haar Cascade with full-image fallback
- Classical baseline: HOG + Linear SVM
- Deep learning model: custom Conv2D CNN from scratch
- Attention: lightweight Squeeze-and-Excitation blocks trained from scratch
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix, ROC curve, AUC, and training curves

## Compute

The full experiment was run on Kaggle Notebook GPU runtime using a Tesla T4 x2 environment.

- Accelerator setting: `NvidiaTeslaT4`
- Training command: `bash run_kaggle.sh`
- Full run duration: approximately **1 hour 31 minutes**
- Run window: `2026-05-26 15:49 UTC` to `2026-05-26 17:20 UTC`
- Max epochs per CNN scenario: `50`
- Early stopping: enabled

## Results

The final model was selected by best validation loss, not by test-set score.

Final selected model:

```text
Scenario     : T2_cnn_enhanced_full_se_aug_wider
Model        : custom Conv2D CNN from scratch
Filters      : [48, 96, 160, 256]
SE blocks    : enabled
Optimizer    : AdamW
Learning rate: 0.0007
Batch size   : 32
Augmentation : enabled
Val loss     : 0.0285
Val accuracy : 0.9920
```

Test-set performance of the selected model:

```text
Accuracy : 0.9833
Precision: 0.9894
Recall   : 0.9774
F1-score : 0.9834
AUC      : 0.9985
```

Best test F1-score scenario:

```text
Scenario : F_cnn_enhanced_full_se_aug
Accuracy : 0.9850
F1-score : 0.9851
AUC      : 0.9982
```

Classical baseline:

```text
HOG + SVM without enhancement
Accuracy : 0.8044
F1-score : 0.8066
AUC      : 0.8815
```

## Scenario Summary

| Scenario | Model | Accuracy | F1-score | AUC |
| --- | --- | ---: | ---: | ---: |
| A | HOG + SVM, plain full image | 0.8044 | 0.8066 | 0.8815 |
| B | HOG + SVM, enhanced full image | 0.7877 | 0.7862 | 0.8768 |
| C | CNN, plain full image, SE, no augmentation | 0.9833 | 0.9834 | 0.9981 |
| D | CNN, enhanced full image, SE, no augmentation | 0.9833 | 0.9834 | 0.9979 |
| E | CNN, enhanced face crop, SE, no augmentation | 0.9780 | 0.9781 | 0.9968 |
| F | CNN, enhanced full image, SE, augmentation | 0.9850 | 0.9851 | 0.9982 |
| G | CNN, enhanced full image, no SE, augmentation | 0.9806 | 0.9807 | 0.9974 |
| T1 | CNN, enhanced, SE, augmentation, LR 5e-4 | 0.9744 | 0.9747 | 0.9973 |
| T2 | CNN wider, enhanced, SE, augmentation | 0.9833 | 0.9834 | 0.9985 |
| T3 | CNN, enhanced, SE, augmentation, Adam, batch 64 | 0.9833 | 0.9833 | 0.9972 |

## Repository Structure

```text
.
├── face_mask_from_scratch_cv.ipynb              # full experiment notebook
├── face_mask_from_scratch_cv_earlystop.ipynb    # faster early-stop notebook
├── kernel-metadata.json                         # Kaggle metadata for full run
├── kernel-metadata-earlystop.json               # Kaggle metadata for early-stop run
├── run_kaggle.sh                                # full run launcher
├── run_kaggle_earlystop.sh                      # early-stop launcher
├── requirements.txt                             # local/Kaggle CLI helper dependencies
└── outputs/
    ├── metrics.json                  # full experiment metrics
    ├── scenario_comparison.png
    ├── preprocessing_examples.png
    ├── confusion_matrix_*.png
    ├── roc_curve_*.png
    └── training_history_*.png
```

Model checkpoints (`*.keras`), Kaggle logs, and report files are intentionally excluded from git to keep the repository lightweight.

## Reproduction

Install local helper dependencies:

```bash
pip install -r requirements.txt
```

Make sure Kaggle credentials are available:

```bash
~/.kaggle/kaggle.json
```

Run the full experiment on Kaggle:

```bash
bash run_kaggle.sh
```

The script pushes the notebook, polls the Kaggle kernel status, and downloads outputs into `outputs/` when the run completes.

For the faster early-stop version:

```bash
bash run_kaggle_earlystop.sh
```

The early-stop notebook keeps the same from-scratch rule but focuses on the best full-run architecture (`T2`) and uses stricter stopping:

- `EarlyStopping(patience=3, min_delta=1e-3, start_from_epoch=8)`
- `ReduceLROnPlateau(patience=2)`
- target-stop callback when `val_loss <= 0.030` and `val_accuracy >= 0.990`
- default architecture: filters `[48, 96, 160, 256]`, SE blocks enabled, AdamW, learning rate `7e-4`, augmentation enabled

## Notes

- The dataset is not committed to this repository.
- The report files are not committed yet.
- All CNN weights are learned from the face mask dataset itself.
- The selected final model is based on validation loss to avoid choosing a model using the test set.
