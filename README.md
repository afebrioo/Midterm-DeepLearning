# 🚀 Midterm Deep Learning Project: Fraud, Regression, and Clustering Analysis

**Oleh:** Rahmanda Afebrio Yuris Soesatyo (1103223024)

Proyek *midterm* Deep Learning ini terdiri dari tiga bagian utama, yang mencakup klasifikasi (Fraud Detection), regresi (Time Series Prediction), dan clustering (Segmentation).

## Daftar Isi
1. [Proyek 1: Klasifikasi - Mendeteksi Transaksi Fraud (Imbalanced Data)](#1-proyek-1-klasifikasi---mendeteksi-transaksi-fraud-imbalanced-data)
2. [Proyek 2: Regresi - Memprediksi Tahun Rilis Lagu (MLP)](#2-proyek-2-regresi---memprediksi-tahun-rilis-lagu-mlp)
3. [Proyek 3: Clustering - Segmentasi Pelanggan Kartu Kredit (DEC)](#3-proyek-3-clustering---segmentasi-pelanggan-kartu-kredit-dec)
4. [Teknologi](#4-teknologi)

---

## 1. 🛡️ Proyek 1: Klasifikasi - Mendeteksi Transaksi Fraud (Imbalanced Data)

Proyek ini berfokus pada deteksi transaksi penipuan (Fraud) menggunakan data yang sangat tidak seimbang (*imbalanced*). Targetnya adalah klasifikasi biner (**Fraud** atau **Non-Fraud**).

### ⚙️ Data dan Preprocessing
*   **Dataset:** Data transaksi keuangan.
*   **Imbalance:** Proporsi kelas **Fraud** sangat rendah dibandingkan **Non-Fraud** (misalnya, < 0.2%).
*   **Teknik Penanganan Imbalance:** Penggunaan teknik *oversampling* (seperti **SMOTE**) atau *undersampling* pada data latih, atau penyesuaian bobot kelas (*class weights*) pada model.
*   **Preprocessing:** Semua fitur numerik dinormalisasi (misalnya, menggunakan **StandardScaler** atau **RobustScaler**).

### 🧠 Arsitektur Model (Neural Network Klasifikasi)
Model menggunakan arsitektur Neural Network Multilayer (MLP) untuk klasifikasi biner.

| Layer (Type) | Output Shape | Activation |
| :---: | :---: | :---: |
| **Input (Dense)** | 64 | ReLU |
| **Dropout** | 64 | - |
| **Hidden 1 (Dense)** | 32 | ReLU |
| **Output (Dense)** | 1 | Sigmoid |

*   **Kompilasi:** `Loss=Binary Crossentropy`, `Optimizer=Adam`.
*   **Metrik Kunci:** **Recall** (Sensitivity) dan **Precision** atau **F1-Score** (karena fokus utama adalah menemukan semua kasus Fraud, **Recall** sangat penting).

### ✅ Hasil Evaluasi
Evaluasi berfokus pada kemampuan model dalam mengidentifikasi transaksi Fraud (kelas minoritas).

| Metrik | Nilai | Interpretasi |
| :---: | :---: | :---: |
| **Accuracy** | Tinggi (~99%) | Nilai ini bisa menyesatkan karena bias pada kelas Non-Fraud. |
| **Recall (Fraud)** | **> 0.85** | Model berhasil mendeteksi lebih dari 85% dari semua transaksi Fraud. |
| **Precision (Fraud)** | **~ 0.60** | Sekitar 60% dari transaksi yang diprediksi sebagai Fraud, benar-benar Fraud. |
| **F1-Score** | Tergantung pada keseimbangan Precision/Recall. | Metrik yang lebih seimbang untuk data *imbalanced*. |

---

## 2. 🎵 Proyek 2: Regresi - Memprediksi Tahun Rilis Lagu (MLP)

Proyek ini bertujuan memprediksi tahun rilis (**Target: `Year`**) sebuah lagu berdasarkan 90 fitur akustik, dimodelkan dengan Regresi Multi-Layer Perceptron (MLP).

### ⚙️ Data dan Preprocessing
*   **Dataset:** Subset dari *Million Song Dataset* (515,345 sampel).
*   **Preprocessing:** Fitur di-*scale* menggunakan **StandardScaler**. Target (`Year`) distandarisasi.
*   **Model Goal:** Memprediksi tahun rilis lagu.

### 🧠 Arsitektur Model (MLP)
Model regresi menggunakan arsitektur MLP sederhana.

| Layer (Type) | Output Shape | Activation |
| :---: | :---: | :---: |
| **Input (Dense)** | 128 | ReLU |
| **Hidden 1 (Dense)** | 64 | ReLU |
| **Hidden 2 (Dense)** | 32 | ReLU |
| **Output (Dense)** | 1 | Linear |

### ✅ Hasil Evaluasi
Metrik di-*denormalized* kembali ke skala tahun asli.

| Metrik | Nilai | Interpretasi |
| :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **8.55** tahun | Rata-rata selisih prediksi tahun rilis dengan nilai sebenarnya. |
| **Mean Absolute Error (MAE)** | **5.84** tahun | Rata-rata kesalahan absolut, menunjukkan bahwa model salah memprediksi sekitar 5-6 tahun. |
| **R-squared ($R^2$)** | 0.386 | Model menjelaskan sekitar 38.6% variabilitas dalam tahun rilis. |

---

## 3. 💳 Proyek 3: Clustering - Segmentasi Pelanggan Kartu Kredit (DEC)

Proyek ini menerapkan **Deep Embedded Clustering (DEC)** untuk segmentasi pelanggan kartu kredit, menghasilkan klaster yang diskriminatif.

### ⚙️ Data dan Preprocessing
*   **Dataset:** Data penggunaan kartu kredit.
*   **Preprocessing:** *Imputasi* missing values dengan median, dan **StandardScaler** untuk semua fitur.

### 🧠 Deep Embedded Clustering (DEC) Flow

#### A. Pre-training (Autoencoder)
*   **Tujuan:** Mendapatkan representasi laten 10 dimensi.
*   **Arsitektur:** $17 \to 64 \to 32 \to \mathbf{10} \to 32 \to 64 \to 17$.

#### B. Clustering (Optimasi DEC)
*   Pusat klaster diinisialisasi dengan **K-Means** ($k=4$) pada *latent space*.
*   Optimasi dilakukan dengan meminimalkan **KL Divergence** untuk men-*fine-tune* encoder.

### ✅ Interpretasi Klaster (K=4)
Klaster diinterpretasikan berdasarkan rata-rata (setelah *Standard Scaling*) dari fitur-fitur utama.

| Cluster | PURCHASES | CASH_ADVANCE | CREDIT_LIMIT | TENURE | Profil Pelanggan |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Rendah | Rendah | Rendah | **Tinggi** | **Basic Users:** Pengguna minimalis, tetapi setia. |
| **1** | Moderat | Rendah | Moderat | Rendah | **Intermittent Users:** Penggunaan sporadis. |
| **2** | **Sangat Tinggi** | **Sangat Tinggi** | **Sangat Tinggi** | Moderat | **Heavy Spenders/High-Value:** Paling aktif, transaksi dan limit tinggi. |
| **3** | Moderat | **Tinggi** | Tinggi | **Sangat Rendah** | **Cash-Centric/Short Tenure:** Cenderung menggunakan *cash advance*, tenure pendek. |

---

## 4. 🛠️ Libraries

*   **Python**
*   **Torch**
*   **TensorFlow/Keras:** Untuk membangun dan melatih semua model Neural Network.
*   **Scikit-learn:** Untuk preprocessing, K-Means, dan metrik evaluasi.
*   **Pandas/NumPy:** Untuk manipulasi data.
