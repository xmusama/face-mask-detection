# Metrics Report

## Ringkasan Run

| Item | Value |
| --- | --- |
| Dataset | omkargurav/face-mask-dataset |
| TensorFlow | 2.19.0 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Best CNN scenario | F_cnn_enhanced_full_se_aug |
| Best test F1 scenario | T2_cnn_enhanced_full_se_aug_wider |

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
| C_cnn_plain_full_se_no_aug | CNN tanpa enhancement | Custom CNN Conv2D from scratch | 0.9692 | 0.9770 | 0.9617 | 0.9693 | 0.9945 | 14 |
| D_cnn_enhanced_full_se_no_aug | Pengaruh enhancement | Custom CNN Conv2D from scratch | 0.9727 | 0.9875 | 0.9583 | 0.9726 | 0.9969 | 12 |
| F_cnn_enhanced_full_se_aug | Pengaruh augmentation | Custom CNN Conv2D from scratch | 0.9718 | 0.9755 | 0.9687 | 0.9721 | 0.9965 | 21 |
| G_cnn_enhanced_full_no_se_aug | Ablation SE block | Custom CNN Conv2D from scratch | 0.9604 | 0.9601 | 0.9617 | 0.9609 | 0.9926 | 8 |
| T1_cnn_enhanced_full_se_aug_lr5e4 | Tuning learning rate | Custom CNN Conv2D from scratch | 0.9559 | 0.9730 | 0.9391 | 0.9558 | 0.9896 | 7 |
| T2_cnn_enhanced_full_se_aug_wider | Tuning jumlah filter dan dropout | Custom CNN Conv2D from scratch | 0.9815 | 0.9894 | 0.9739 | 0.9816 | 0.9980 | 21 |
| T3_cnn_enhanced_full_se_aug_adam | Tuning optimizer dan batch size | Custom CNN Conv2D from scratch | 0.9683 | 0.9607 | 0.9774 | 0.9690 | 0.9974 | 15 |
