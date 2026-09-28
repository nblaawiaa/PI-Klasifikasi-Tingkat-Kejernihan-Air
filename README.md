# Klasifikasi Tingkat Kejernihan Air Menggunakan CNN dan Random Forest

Aplikasi berbasis web untuk mengklasifikasikan citra air menjadi dua kelas, yaitu **Jernih** dan **Keruh** menggunakan kombinasi **Convolutional Neural Network (CNN)** dan **Random Forest**.

## Tentang Project

Project ini merupakan implementasi dari Penulisan Ilmiah dengan judul:

**Klasifikasi Tingkat Kejernihan Air Menggunakan CNN dan Random Forest**

CNN digunakan untuk mengekstraksi fitur visual dari citra, kemudian fitur tersebut digunakan oleh Random Forest untuk melakukan klasifikasi akhir.

Model menghasilkan:
- Kelas **Jernih** atau **Keruh**
- Nilai *confidence*
- Probabilitas masing-masing kelas
- Persentase kekeruhan berdasarkan probabilitas kelas Keruh

## Dataset

Dataset terdiri dari dua kelas:

- Jernih: 500 citra
- Keruh: 500 citra
- Total: 1.000 citra

Citra yang digunakan memiliki format JPG, JPEG, PNG, dan WEBP.

## Model

### CNN

CNN digunakan sebagai *feature extractor* untuk mengambil fitur visual dari citra.

Input citra:
- Ukuran: 224 × 224 piksel
- Format: RGB
- Normalisasi: nilai piksel dibagi 255

CNN menghasilkan fitur dari lapisan Flatten yang kemudian digunakan sebagai input Random Forest.

### Random Forest

Random Forest digunakan sebagai classifier akhir.

Konfigurasi yang digunakan:

- `n_estimators = 200`
- `max_depth = 20`
- `min_samples_split = 4`
- `min_samples_leaf = 2`
- `max_features = sqrt`
- `class_weight = balanced`

## Hasil

Pada data uji, model CNN secara mandiri memperoleh akurasi sebesar **91%**, sedangkan kombinasi CNN dan Random Forest memperoleh akurasi sebesar **96%**.

Confusion matrix model CNN + Random Forest:

| | Prediksi Jernih | Prediksi Keruh |
|---|---:|---:|
| Jernih | 47 | 3 |
| Keruh | 1 | 49 |

## Aplikasi

Aplikasi dibuat menggunakan **Streamlit**.

Alur aplikasi:

1. Pengguna mengunggah citra.
2. Citra diproses dan diubah ke ukuran 224 × 224 piksel.
3. CNN melakukan ekstraksi fitur.
4. Random Forest melakukan klasifikasi.
5. Aplikasi menampilkan hasil klasifikasi, confidence, probabilitas kelas, dan persentase kekeruhan.