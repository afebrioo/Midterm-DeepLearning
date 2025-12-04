# 📉 Deep Learning Regression: Memprediksi Tahun Produksi Musik

**Nama/NIM:** Rahmanda Afebrio Yuris Soesatyo - 1103223024

Proyek ini menggunakan **Multi-Layer Perceptron (MLP)** yang dibangun dengan **TensorFlow/Keras** untuk menyelesaikan tugas regresi: memprediksi tahun rilis (**Target: `Year`**) sebuah lagu berdasarkan 90 fitur akustik dan properti musik lainnya. Dataset yang digunakan adalah subset dari *Million Song Dataset*.

---

## 1. ⚙️ Data Preparation & Preprocessing

Dataset ini memiliki tantangan utama berupa *missing header*, data yang besar, dan target regresi yang berskala besar (tahun).

### 📥 Pemuatan Data
*   Dataset dimuat dari file CSV tanpa *header*, sehingga nama kolom disematkan secara manual (`target` (Year) dan `feature_1` hingga `feature_90`).
*   Total sampel data yang digunakan: **515,345 baris** dan **91 kolom**.

### ⚖️ Target Standardization
Untuk membantu konvergensi model Deep Learning yang lebih cepat, variabel target (`Year`) distandarisasi (*denormalized*):
$$y_{\text{scaled}} = \frac{y - \mu_{\text{target}}}{\sigma_{\text{target}}}$$

*   Mean Target ($\mu_{\text{target}}$): **1998.40**
*   Standard Deviation Target ($\sigma_{\text{target}}$): **10.93**

### 🧩 Feature Preprocessing Pipeline
Semua 90 fitur adalah numerik. Preprocessing diterapkan menggunakan `sklearn.pipeline` dan `sklearn.compose` pada data latih dan uji:
1.  **Imputasi:** Mengisi nilai *missing* (`NaN`) menggunakan **median**.
2.  **Scaling:** Normalisasi fitur menggunakan **StandardScaler** (Z-Score Normalization).

### 📊 Pembagian Data
*   Data dibagi menjadi *Training Set* (80%) dan *Test Set* (20%).
*   **Training Set Size:** 412,276 rows
*   **Test Set Size:** 103,069 rows

---

## 2. 🧠 Arsitektur Model (MLP)

Model yang digunakan adalah Multi-Layer Perceptron (MLP) yang sederhana namun efektif untuk tugas regresi:

### Arsitektur Jaringan
| Layer (Type) | Output Shape | Activation | Regularization |
| :---: | :---: | :---: | :---: |
| **Input (Dense)** | 128 | ReLU | Dropout(0.2) |
| **Hidden 1 (Dense)** | 64 | ReLU | - |
| **Hidden 2 (Dense)** | 32 | ReLU | - |
| **Output (Dense)** | 1 | Linear | - |

*   **Total Parameters:** 22,017

### 🚀 Kompilasi Model
*   **Optimizer:** `Adam` (Learning Rate: 0.001)
*   **Loss Function:** Mean Squared Error (`mse`)
*   **Metrics:** Mean Absolute Error (`mae`), Mean Squared Error (`mse`)

---

## 3. ⏱️ Pelatihan dan Regularisasi

Model dilatih dengan strategi *early stopping* untuk mencegah *overfitting* dan meningkatkan efisiensi.

*   **Epochs Max:** 100
*   **Batch Size:** 64
*   **Validation Split:** 10% dari data training.
*   **Early Stopping:** Monitor `val_loss`, `patience=5`.

### Hasil Pelatihan
*   Pelatihan dihentikan pada **Epoch 25** (Epoch terbaik: 20).
*   **Training Duration:** 158.89 seconds.
*   **Best Validation Loss (Scaled MSE):** 0.6036 (dicapai pada Epoch 20).

---

## 4. ✅ Evaluasi Performa

Evaluasi dilakukan pada *Test Set* yang belum pernah dilihat model, menggunakan prediksi yang telah di-*denormalized* kembali ke skala tahun yang asli.

| Metrik Evaluasi | Nilai | Interpretasi |
| :---: | :---: | :---: |
| **MSE (Mean Squared Error)** | **73.0717** | Rata-rata kuadrat kesalahan dalam satuan tahun². |
| **RMSE (Root Mean Squared Error)** | **8.5482** | Rata-rata kesalahan prediksi (dalam tahun). Model salah prediksi rata-rata $\approx 8.5$ tahun. |
| **MAE (Mean Absolute Error)** | **5.8431** | Rata-rata kesalahan absolut (dalam tahun). |
| **R-squared ($R^2$)** | **0.3860** | Sekitar 38.6% dari variabilitas tahun rilis dapat dijelaskan oleh model. |