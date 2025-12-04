# 🤖 Deep Learning Fraud Detection (Midterm Project)

**Nama/NIM:** Rahmanda Afebrio Yuris Soesatyo - 1103223024

Proyek ini bertujuan untuk membangun dan melatih **Multi-Layer Perceptron (MLP)** menggunakan PyTorch untuk memprediksi transaksi *fraud* (`isFraud` = 1) pada data transaksi yang sangat tidak seimbang (*highly imbalanced*). Proses ini mencakup *data preprocessing* yang agresif, *feature engineering*, penanganan *class imbalance* menggunakan **SMOTE**, dan pelatihan model dengan teknik-teknik *deep learning* modern.

---

## 1. 🛠️ Persiapan dan Lingkungan

Proyek ini menggunakan **PyTorch** dan memanfaatkan akselerasi **CUDA (GPU)**.

*   **PyTorch Version:** 2.6.0+cu124
*   **CUDA Available:** True
*   **Device:** NVIDIA GeForce RTX 3050 Laptop GPU

---

## 2. ⚙️ Data Preprocessing & Feature Engineering

Dilakukan langkah *preprocessing* dan reduksi fitur yang **agresif** (disebabkan oleh keterbatasan RAM) untuk memampatkan data dimensi tinggi, sambil tetap mempertahankan sinyal penting.

### Feature Engineering
1.  **Transformasi Logaritmik:** Menerapkan $log(1+x)$ pada fitur-fitur `amount` untuk mengurangi *skewness*.
2.  **Fitur Temporal:** Ekstraksi jam, hari, dan minggu dari `TransactionDT`.
3.  **Rasio Transaksi:** Menghitung rasio `TransactionAmt` terhadap rata-rata *amount* per `card1` dan `card2`.
4.  **Agregasi & Frekuensi:** Menghitung *count* per kolom `card1`, `card2`, `card3`, dan `card4`.

### Reduksi Fitur Agresif
1.  **Pembersihan:** Menghapus kolom konstan dan kolom dengan $>90\%$ *missing values*.
2.  **Encoding:** Mengubah fitur kategorikal (`object`) menjadi numerik menggunakan `LabelEncoder`.
3.  **Imputasi:** Mengisi semua nilai NaN dengan **median** (menggunakan *numpy* untuk efisiensi).
4.  **Variance Filtering:** Memilih hanya **75%** fitur dengan varians tertinggi (Total fitur tersisa: **303**).
5.  **Scaling:** Data dinormalisasi menggunakan `StandardScaler`.

---

## 3. ⚖️ Penanganan Class Imbalance (SMOTE)

Data *training* asli memiliki rasio *fraud* hanya **3.499%**. Untuk menanggulangi ketidakseimbangan ini, digunakan:

*   **SMOTE (Synthetic Minority Over-sampling Technique):** Diterapkan pada data *training* (80% split) untuk menyeimbangkan kelas.
    *   *Train set* awal: 472,432 sampel.
    *   *Train set* setelah SMOTE: **911,804 sampel** (seimbang).
*   **Validation Set:** Menggunakan data asli yang terstratifikasi (20% split) untuk evaluasi performa yang realistis.

---

## 4. 🧠 Arsitektur dan Tuning Model (MLP)

Model **ImprovedFraudMLP** adalah *deep* MLP dengan fokus pada regularisasi dan optimasi:

### Arsitektur Model (303 Input Features)
| Layer | Output Dim | Teknik |
| :---: | :---: | :---: |
| **FC1** | 256 | ReLU, BatchNorm1d, Dropout(0.3) |
| **FC2** | 128 | ReLU, BatchNorm1d, Dropout(0.3) |
| **FC3** | 64 | ReLU, BatchNorm1d, Dropout(0.2) |
| **FC4** | 1 | Sigmoid |

### Strategi Pelatihan
*   **Optimizer:** `AdamW` (dengan *weight decay* $1e^{-5}$).
*   **Scheduler:** `CosineAnnealingLR` untuk jadwal *learning rate* yang dinamis.
*   **Batch Size:** 2048.
*   **Regularization:** Penggunaan `BatchNorm1d` dan `Dropout` di setiap lapisan tersembunyi.
*   **Pencegahan Overfitting:** `ImprovedEarlyStopping` dengan *patience* 5 dan *min\_delta* 0.0001.

### Hasil Pelatihan
*   **Best Validation AUC:** **0.933618**
*   **Epoch Akhir:** 24 (dihentikan oleh *Early Stopping*).

---

## 5. 💾 Hasil Prediksi

Prediksi probabilitas *fraud* pada *test set* dihasilkan menggunakan model PyTorch terbaik.

| Statistik Prediksi | Nilai |
| :---: | :---: |
| **Mean** | 0.060030 |
| **Median** | 0.008307 |
| **Std** | 0.163962 |
| **Min/Max** | 0.000000 / 1.000000 |

Model PyTorch terbaik disimpan sebagai `best_fraud_model_pytorch.pth`.

---