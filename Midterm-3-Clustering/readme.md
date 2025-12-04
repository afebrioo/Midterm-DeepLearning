# 💳 Deep Embedded Clustering (DEC) - Analisis Segmentasi Kartu Kredit

**Nama/NIM:** Rahmanda Afebrio Yuris Soesatyo - 1103223024

Proyek ini bertujuan untuk melakukan segmentasi pelanggan (clustering) pada data penggunaan kartu kredit menggunakan metode Deep Embedded Clustering (DEC). DEC menggabungkan **Autoencoder** (untuk *dimensionality reduction* dan *feature learning*) dengan optimasi berbasis **Kullback-Leibler (KL) Divergence** (untuk *clustering*), menghasilkan representasi fitur yang lebih diskriminatif untuk pengelompokan.

---

## 1. ⚙️ Data Preparation & Preprocessing

### 📥 Dataset
*   Dataset: **`clusteringmidterm.csv`** (Credit Card Customer Data)
*   Total sampel: **8,950** baris dan **18** kolom.
*   Fitur utama mencakup saldo, frekuensi pembelian, transaksi tunai, batas kredit, pembayaran, dsb.

### 🧹 Preprocessing
1.  **Drop Kolom:** Kolom `CUST_ID` dihapus karena tidak relevan untuk clustering.
2.  **Missing Value Imputation:** Nilai yang hilang (`NaN`) pada kolom `CREDIT_LIMIT` (1 nilai) dan `MINIMUM_PAYMENTS` (313 nilai) diisi menggunakan **median** dari masing-masing kolom.
3.  **Feature Scaling:** Semua fitur diskalakan menggunakan **StandardScaler** (Z-Score Normalization) untuk memastikan semua fitur memiliki bobot yang sama dalam proses clustering.

---

## 2. 🧠 Deep Embedded Clustering (DEC)

DEC terdiri dari dua tahap: *Pre-training* (Autoencoder) dan *Fine-tuning* (Optimasi Clustering).

### A. Autoencoder (Pre-training)

Sebuah Autoencoder digunakan untuk memampatkan data (17 dimensi) menjadi representasi laten berdimensi rendah (**Latent Dimension: 10**) yang menangkap informasi penting.

#### Arsitektur Autoencoder
| Layer (Type) | Output Shape | Activation | Purpose |
| :---: | :---: | :---: | :---: |
| **Input** | 17 | - | Input data asli (scaled) |
| **Encoder 1** | 64 | ReLU | |
| **Encoder 2** | 32 | ReLU | |
| **Latent/Bottleneck** | **10** | ReLU | Representasi fitur tereduksi |
| **Decoder 1** | 32 | ReLU | |
| **Decoder 2** | 64 | ReLU | |
| **Output** | 17 | Linear | Rekonstruksi input |

*   **Kompilasi:** `optimizer='adam'`, `loss='mse'` (Mean Squared Error).
*   **Pelatihan:** 50 epochs, `batch_size=256`.

### B. Inisialisasi K-Means
Representasi laten (**latent space**) yang dihasilkan oleh *Encoder* (**`latent_repr`**) kemudian digunakan untuk menginisialisasi pusat cluster (**Cluster Centers**) menggunakan algoritma **K-Means** tradisional.

#### Penentuan Jumlah Cluster (Elbow Method)
Metode *Elbow* pada ruang laten menunjukkan titik belok yang signifikan pada $k=4$, sehingga **jumlah cluster** (**k**) ditetapkan menjadi **4**.

$$k = 4$$

### C. Optimasi DEC (Fine-tuning)

DEC melanjutkan pelatihan *Encoder* dengan meminimalkan **Kullback-Leibler (KL) Divergence** antara distribusi probabilitas *soft assignment* (**Q**) dan distribusi target yang diidealkan (**P**).

*   **Tujuan:** Mengasah *latent space* sehingga titik data lebih dekat ke pusat cluster yang sesuai, menghasilkan cluster yang lebih murni.
*   **Metode:** Iteratif (3000 iterasi), menggunakan *gradient descent* (Adam optimizer) pada bobot *Encoder*.

---

## 3. ✅ Evaluasi dan Interpretasi Hasil

Setelah optimasi DEC, klaster akhir (`final_clusters`) didapatkan melalui *soft assignment* pada ruang laten akhir (`final_z`).

### A. Visualisasi Klaster
Proyeksi 2D menggunakan **t-SNE** menunjukkan klaster yang **terpisah dengan baik** di ruang laten.



### B. Ringkasan Profil Klaster

Tabel di bawah menunjukkan rata-rata (setelah *Standard Scaling*) dari fitur-fitur utama untuk setiap klaster. Nilai positif menunjukkan rata-rata yang lebih tinggi dari keseluruhan populasi, dan sebaliknya.

| Cluster | BALANCE | PURCHASES | ONEOFF_PURCHASES | CASH_ADVANCE | PURCHASES_TRX | CREDIT_LIMIT | PAYMENTS | TENURE | **Profil Utama** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | -0.20 | -0.26 | -0.24 | -0.30 | -0.24 | -0.28 | -0.29 | **+0.31** | Pengguna **Basic** (Saldo & transaksi rendah, tenure panjang) |
| **1** | -0.16 | -0.14 | -0.13 | -0.05 | -0.16 | -0.05 | -0.14 | -0.25 | Pengguna **Awal/Intermittent** (Saldo, transaksi, limit moderat/rendah) |
| **2** | **+1.32** | **+2.02** | **+1.79** | **+1.23** | **+1.97** | **+1.31** | **+1.92** | +0.04 | **High-Value/Heavy Spenders** (Semua metrik sangat tinggi) |
| **3** | +0.42 | +0.21 | +0.22 | +0.60 | +0.20 | +0.46 | +0.41 | **-0.62** | Pengguna **Cash-Centric/Tenure Pendek** (Cash Advance tinggi, pembelian moderat, tenure sangat pendek) |

### C. Analisis Metrik Tradisional (K-Means pada Data Asli)

Skor Siluet K-Means pada **data asli** (sebelum DEC) menunjukkan bahwa kualitas *clustering* tanpa *deep feature learning* kurang optimal:

| K | Silhouette Score |
| :-: | :-: |
| 2 | 0.2795 |
| 3 | 0.2067 |
| **4** | **0.1665** |
| 5 | 0.1926 |

*Catatan: Meskipun Siluet Score K-Means pada data asli lebih tinggi pada k=2, pemisahan visual klaster dan interpretasi profil DEC pada k=4 memberikan hasil segmentasi yang lebih bermakna secara bisnis.*

---

Apakah Anda ingin saya memberikan detail lebih lanjut mengenai profil setiap klaster atau melihat hasil evaluasi yang lain?
