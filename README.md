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

The current execution target is Google Colab GPU runtime. A Colab-ready runner is provided at `source/00_colab_run_all.ipynb`.

- Runtime: Google Colab GPU
- Dataset source: Kaggle Face Mask Dataset (`omkargurav/face-mask-dataset`)
- Runner: `source/00_colab_run_all.ipynb`
- Output folder inside Colab: `/content/results`
- Max epochs per CNN scenario in the Colab runner: `25`
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
source/         # Split experiment notebooks and Colab runner
scripts/        # Reserved for the Streamlit app
automation/     # Kaggle run and polling helpers
results/
  figures/      # Scenario comparison, preprocessing examples, confusion matrices, ROC curves, training curves
  metrics/      # Full experiment metrics JSON
  models/       # Generated model checkpoints, kept out of git
requirements.txt
kernel-metadata.json
README.md
```

Notebook order:

```text
source/00_colab_run_all.ipynb  # Recommended for Colab execution
source/01_eda.ipynb
source/02_preprocessing.ipynb
source/03_classical.ipynb
source/04_cnn.ipynb
source/05_evaluation.ipynb
```

The first notebook keeps EDA separated by method: class distribution, raw image samples, image resolution, brightness/contrast, noise/blur, edge and HOG visualization, pose/background variation, and Haar Cascade face detection coverage. The 70/15/15 data split is placed at the final cell of `01_eda.ipynb` after the original-image EDA is complete.

For modular execution, run the notebooks in order in the same kernel/session. The modeling notebooks only train and store models; all test-set metrics, ROC curves, confusion matrices, scenario comparison, and final export are handled in `05_evaluation.ipynb`.

Model checkpoints (`*.keras`), Kaggle logs, and report files are intentionally excluded from git to keep the repository lightweight.
The Streamlit-ready final CNN checkpoint is included as `results/models/face_mask_custom_cnn_from_scratch_best.keras` so prediction can run after remote deployment.

## Reproduction

Install local helper dependencies:

```bash
pip install -r requirements.txt
```

Recommended Colab flow:

```text
1. Open source/00_colab_run_all.ipynb in Google Colab.
2. Select Runtime -> Change runtime type -> GPU.
3. Run all cells.
4. Upload kaggle.json when prompted.
5. Download the generated /content/face_mask_results.zip.
```

For local or Kaggle-style execution, make sure Kaggle credentials are available:

```bash
~/.kaggle/kaggle.json
```

Run the modular notebooks sequentially in one session:

```text
01_eda -> 02_preprocessing -> 03_classical -> 04_cnn -> 05_evaluation
```

Generated figures are organized under `results/figures/`, metrics under `results/metrics/`, and model files under `results/models/`.

## Streamlit App

Run the local dashboard:

```bash
streamlit run scripts/app.py
```

The app reads metrics and figures from `results/`. Image prediction uses `results/models/face_mask_custom_cnn_from_scratch_best.keras` for the CNN model and the `.joblib` files for HOG-SVM.

Remote deployment settings:

```text
Platform : Streamlit Community Cloud or compatible Streamlit host
Branch   : deployment branch
Main file: scripts/app.py
Python   : runtime.txt
Packages : requirements.txt
```

## Notes

- The dataset is not committed to this repository.
- The report files are not committed yet.
- All CNN weights are learned from the face mask dataset itself.
- The selected final model is based on validation loss to avoid choosing a model using the test set.
