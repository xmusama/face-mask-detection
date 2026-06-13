# Face Mask Detection From Scratch

Pipeline computer vision untuk klasifikasi penggunaan masker wajah secara biner menggunakan dua pendekatan: fitur klasik HOG + SVM dan CNN custom yang dilatih dari awal.

Project ini dibuat untuk praktikum computer vision. Seluruh model deep learning dilatih tanpa pretrained model dan tanpa transfer learning, sehingga tidak menggunakan ResNet, MobileNet, EfficientNet, VGG pretrained, `tf.keras.applications`, atau bobot ImageNet.

## Ringkasan

- Dataset: Kaggle Face Mask Dataset (`omkargurav/face-mask-dataset`)
- Kelas: `with_mask` dan `without_mask`
- Split data: train/validation/test = `70/15/15`
- Ukuran input: `128x128`
- Enhancement dan restoration: CLAHE + Gaussian denoising ringan
- Face crop: OpenCV Haar Cascade dengan fallback ke gambar penuh
- Baseline klasik: HOG + Linear SVM
- Model deep learning: custom Conv2D CNN from scratch
- Attention: Squeeze-and-Excitation block ringan yang juga dilatih dari awal
- Evaluasi: accuracy, precision, recall, F1-score, confusion matrix, ROC curve, AUC, dan training curve

## Komputasi

Target eksekusi utama adalah Google Colab GPU. Notebook runner yang siap dijalankan tersedia di `source/00_colab_run_all.ipynb`.

- Runtime: Google Colab GPU
- Sumber dataset: Kaggle Face Mask Dataset (`omkargurav/face-mask-dataset`)
- Runner: `source/00_colab_run_all.ipynb`
- Folder output di Colab: `/content/results`
- Early stopping: aktif

## Hasil

Model final dipilih berdasarkan validation loss terbaik, bukan berdasarkan skor test set.

Model final:

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

Performa test set model final:

```text
Accuracy : 0.9833
Precision: 0.9894
Recall   : 0.9774
F1-score : 0.9834
AUC      : 0.9985
```

Skenario dengan F1-score test terbaik:

```text
Scenario : F_cnn_enhanced_full_se_aug
Accuracy : 0.9850
F1-score : 0.9851
AUC      : 0.9982
```

Baseline klasik:

```text
HOG + SVM tanpa enhancement
Accuracy : 0.8044
F1-score : 0.8066
AUC      : 0.8815
```

## Ringkasan Skenario

| Scenario | Model | Accuracy | F1-score | AUC |
| --- | --- | ---: | ---: | ---: |
| A | HOG + SVM, gambar penuh tanpa enhancement | 0.8044 | 0.8066 | 0.8815 |
| B | HOG + SVM, gambar penuh dengan enhancement | 0.7877 | 0.7862 | 0.8768 |
| C | CNN, gambar penuh tanpa enhancement, SE, tanpa augmentation | 0.9833 | 0.9834 | 0.9981 |
| D | CNN, gambar penuh dengan enhancement, SE, tanpa augmentation | 0.9833 | 0.9834 | 0.9979 |
| E | CNN, face crop dengan enhancement, SE, tanpa augmentation | 0.9780 | 0.9781 | 0.9968 |
| F | CNN, gambar penuh dengan enhancement, SE, augmentation | 0.9850 | 0.9851 | 0.9982 |
| G | CNN, gambar penuh dengan enhancement, tanpa SE, augmentation | 0.9806 | 0.9807 | 0.9974 |
| T1 | CNN, enhancement, SE, augmentation, LR 5e-4 | 0.9744 | 0.9747 | 0.9973 |
| T2 | CNN lebih lebar, enhancement, SE, augmentation | 0.9833 | 0.9834 | 0.9985 |
| T3 | CNN, enhancement, SE, augmentation, Adam, batch 64 | 0.9833 | 0.9833 | 0.9972 |

## Struktur Repository

```text
source/         # Notebook eksperimen dan runner Colab/Kaggle/lokal
scripts/        # Streamlit app dan script pendukung
automation/     # Helper untuk menjalankan/polling Kaggle
results/
  figures/      # Grafik EDA, preprocessing, confusion matrix, ROC, training curve
  metrics/      # Metrik eksperimen dalam JSON dan Markdown
  models/       # Model final CNN dan model HOG-SVM untuk Streamlit
requirements.txt
runtime.txt
kernel-metadata.json
README.md
```

Urutan notebook modular:

```text
source/00_colab_run_all.ipynb  # Direkomendasikan untuk eksekusi Colab
source/01_eda.ipynb
source/02_preprocessing.ipynb
source/03_classical.ipynb
source/04_cnn.ipynb
source/05_evaluation.ipynb
```

Notebook `01_eda.ipynb` memisahkan analisis menjadi beberapa bagian: distribusi kelas, sampel gambar asli, resolusi gambar, brightness/contrast, noise/blur, visualisasi edge dan HOG, variasi pose/background, serta cakupan deteksi wajah Haar Cascade. Split data `70/15/15` ditempatkan di cell terakhir setelah analisis gambar asli selesai.

Notebook modeling hanya berisi training model. Evaluasi test set, confusion matrix, ROC curve, perbandingan skenario, sampel prediksi dengan bounding box, dan export artefak akhir dikerjakan di `05_evaluation.ipynb`.

## Reproduksi

Install dependency:

```bash
pip install -r requirements.txt
```

Alur Colab yang direkomendasikan:

```text
1. Buka source/00_colab_run_all.ipynb di Google Colab.
2. Pilih Runtime -> Change runtime type -> GPU.
3. Jalankan semua cell.
4. Upload kaggle.json ketika diminta.
5. Download file /content/face_mask_results.zip yang dihasilkan.
```

Untuk eksekusi lokal atau Kaggle-style, pastikan kredensial Kaggle tersedia:

```bash
~/.kaggle/kaggle.json
```

Jika memakai notebook modular, jalankan berurutan dalam satu kernel/session:

```text
01_eda -> 02_preprocessing -> 03_classical -> 04_cnn -> 05_evaluation
```

Output gambar tersimpan di `results/figures/`, metrik di `results/metrics/`, dan model di `results/models/`.

## Streamlit App

Jalankan dashboard lokal:

```bash
streamlit run scripts/app.py
```

App membaca metrik dan figure dari `results/`. Prediksi gambar menggunakan model CNN `results/models/face_mask_custom_cnn_from_scratch_best.keras` dan model HOG-SVM `.joblib`.

Konfigurasi deployment remote:

```text
Platform : Streamlit Community Cloud atau host Streamlit kompatibel
Branch   : streamlit-dashboard-cnn-remote
Main file: scripts/app.py
Python   : runtime.txt
Packages : requirements.txt
```

Repository GitHub hanya menyimpan source code dan artefak yang diperlukan. Agar dashboard benar-benar bisa diakses publik, branch ini perlu dibuat sebagai app di Streamlit Community Cloud dengan main file `scripts/app.py`.

## Catatan

- Dataset tidak disimpan di repository.
- Model final CNN untuk Streamlit ikut disimpan agar prediksi remote dapat berjalan setelah deployment.
- Checkpoint CNN lain, log runtime, dan file laporan tidak ikut disimpan agar repository tetap ringan.
- Seluruh bobot CNN dipelajari dari dataset face mask, bukan dari pretrained model.
- Model final dipilih berdasarkan validation loss untuk menghindari pemilihan model memakai test set.
