# Metrics Report

## Ringkasan Run

| Item | Value |
| --- | --- |
| Dataset | omkargurav/face-mask-dataset |
| TensorFlow | 2.19.0 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Best CNN scenario | T3_cnn_enhanced_full_se_aug_adam |
| Best test F1 scenario | G_cnn_enhanced_full_no_se_aug |

## Distribusi Kelas

| Class | Count |
| --- | --- |
| with_mask | 3725 |
| without_mask | 3828 |

## Split Data

| Split | Size |
| --- | --- |
| train | 5286 |
| validation | 1132 |
| test | 1135 |

## Hasil Skenario

| Scenario | Purpose | Model Type | Accuracy | Precision | Recall | F1-score | AUC | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_hog_svm_plain_full | Baseline HOG-SVM tanpa enhancement | HOG + LinearSVC | 0.8044 | 0.8080 | 0.8052 | 0.8066 | 0.8815 | - |
| B_hog_svm_enhanced_full | HOG-SVM dengan enhancement CLAHE dan denoise | HOG + LinearSVC | 0.7877 | 0.8025 | 0.7704 | 0.7862 | 0.8768 | - |
| C_cnn_plain_full_se_no_aug | CNN tanpa enhancement | Custom CNN Conv2D from scratch | 0.9692 | 0.9821 | 0.9565 | 0.9692 | 0.9959 | 23 |
| D_cnn_enhanced_full_se_no_aug | Pengaruh enhancement | Custom CNN Conv2D from scratch | 0.8687 | 0.9712 | 0.7635 | 0.8549 | 0.9713 | 10 |
| F_cnn_enhanced_full_se_aug | Pengaruh augmentation | Custom CNN Conv2D from scratch | 0.6555 | 0.9946 | 0.3217 | 0.4862 | 0.9363 | 7 |
| E_cnn_enhanced_crop_se_no_aug | Pengaruh face crop | Custom CNN Conv2D from scratch | 0.9357 | 0.9846 | 0.8870 | 0.9332 | 0.9914 | 14 |
| G_cnn_enhanced_full_no_se_aug | Ablation SE block | Custom CNN Conv2D from scratch | 0.9824 | 0.9843 | 0.9809 | 0.9826 | 0.9974 | 29 |
| T1_cnn_enhanced_full_se_aug_lr5e4 | Tuning learning rate | Custom CNN Conv2D from scratch | 0.9683 | 0.9838 | 0.9530 | 0.9682 | 0.9955 | 31 |
| T2_cnn_enhanced_full_se_aug_wider | Tuning jumlah filter dan dropout | Custom CNN Conv2D from scratch | 0.9463 | 0.9673 | 0.9252 | 0.9458 | 0.9885 | 26 |
| T3_cnn_enhanced_full_se_aug_adam | Tuning optimizer dan batch size | Custom CNN Conv2D from scratch | 0.9815 | 0.9843 | 0.9791 | 0.9817 | 0.9982 | 9 |

## Artefak Utama

| Artifact | Path |
| --- | --- |
| best_keras_model | /kaggle/working/face_mask_custom_cnn_from_scratch_best.keras |
| metrics | /kaggle/working/metrics.json |
| eda_class_distribution | /kaggle/working/eda_class_distribution.png |
| eda_original_samples | /kaggle/working/eda_original_samples.png |
| eda_resolution_aspect_ratio | /kaggle/working/eda_resolution_aspect_ratio.png |
| eda_brightness_contrast | /kaggle/working/eda_brightness_contrast.png |
| eda_noise_blur | /kaggle/working/eda_noise_blur.png |
| eda_edge_hog_examples | /kaggle/working/eda_edge_hog_examples.png |
| eda_pose_background_variation | /kaggle/working/eda_pose_background_variation.png |
| eda_face_detection_coverage | /kaggle/working/eda_face_detection_coverage.png |
| eda_face_detection_examples | /kaggle/working/eda_face_detection_examples.png |
| eda_class_split_distribution | /kaggle/working/eda_class_split_distribution.png |
| class_distribution | /kaggle/working/class_distribution.png |
| preprocessing_examples | /kaggle/working/preprocessing_examples.png |
| scenario_comparison | /kaggle/working/scenario_comparison.png |
