import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(
    page_title="Klasifikasi Kekeruhan Air",
    layout="centered"
)

IMG_SIZE = (224, 224)

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] {
        background-color: #E3F2FD;
        border: 2px dashed #2196F3;
        border-radius: 12px;
        padding: 20px;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #2196F3;
        color: white;
        border-radius: 8px;
        border: none;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #1976D2;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_models():
    # PENTING: cnn_model.h5 disimpan sebagai model UTUH (cnn_model.save(...)),
    # bukan hanya bobot. Jadi kita load modelnya langsung apa adanya, lalu
    # ambil output dari layer 'flatten' -- persis seperti proses ekstraksi
    # fitur di notebook training (lihat: feature_extractor = models.Model(
    # inputs=cnn_model.input, outputs=cnn_model.get_layer('flatten').output)).
    #
    # Cara ini menghindari risiko arsitektur/nama layer yang tidak cocok
    # persis dengan model hasil training, yang sebelumnya menyebabkan
    # ValueError jumlah fitur tidak sesuai dengan rf_model.pkl.
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="nabilaaw/klasifikasi-kejernihan-air",
        filename="cnn_model.h5"
    )

    full_cnn_model = tf.keras.models.load_model(model_path, compile=False)

    flatten_layer = None
    for layer in full_cnn_model.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            flatten_layer = layer
            break

    if flatten_layer is None:
        raise ValueError(
            "Tidak ditemukan layer Flatten di cnn_model.h5. "
            f"Layer yang ada: {[l.name for l in full_cnn_model.layers]}"
        )

    feature_extractor = models.Model(
        inputs=full_cnn_model.input,
        outputs=flatten_layer.output,
        name="CNN_FeatureExtractor"
    )

    rf_model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    return rf_model, label_encoder, feature_extractor


def preprocess_image(pil_img, img_size=IMG_SIZE):
    img = pil_img.convert("RGB")
    img = img.resize(img_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def extract_features(img_array, feature_extractor):
    batch = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(batch, verbose=0)
    return features


def predict_image(pil_img, rf_model, feature_extractor):
    img_array = preprocess_image(pil_img)
    features = extract_features(img_array, feature_extractor)

    proba = rf_model.predict_proba(features)[0]
    prob_jernih = float(proba[0])
    prob_keruh = float(proba[1])

    # Aturan klasifikasi berdasarkan probabilitas kelas Keruh
    label = "Keruh" if prob_keruh >= 0.5 else "Jernih"
    persentase_kekeruhan = round(prob_keruh * 100, 1)

    hasil = {
        "label": label,
        "persentase_kekeruhan": persentase_kekeruhan,
        "confidence": {
            "Jernih": round(prob_jernih * 100, 1),
            "Keruh": round(prob_keruh * 100, 1)
        }
    }
    return hasil


def main():
    with st.sidebar:
        st.header("ℹ️ Tentang Aplikasi")
        st.write(
            "Aplikasi ini dibuat untuk keperluan Penulisan Ilmiah "
            "dengan judul **\"Klasifikasi Tingkat Kekeruhan Air "
            "Menggunakan CNN dan Random Forest Berbasis Web\"**."
        )

        st.markdown("### 🎨 Keterangan Label")
        st.markdown("🔵 **Jernih** — kekeruhan rendah")
        st.markdown("🔴 **Keruh** — kekeruhan tinggi")

        st.markdown("---")
        st.caption("Universitas Gunadarma")

    st.title("Klasifikasi Tingkat Kekeruhan Air")
    st.markdown(
        "**Klasifikasi Tingkat Kekeruhan Air Menggunakan CNN dan "
        "Random Forest Berbasis Web**"
    )
    st.write(
        "Aplikasi ini mengklasifikasikan tingkat kekeruhan air "
        "(Jernih atau Keruh) berdasarkan gambar yang diunggah. "
        "Model menggunakan CNN sebagai ekstraksi fitur dan "
        "Random Forest sebagai klasifikasi akhir."
    )
    st.markdown("---")

    # label_encoder dimuat oleh load_models() tetapi tidak dipakai di sini
    rf_model, _, feature_extractor = load_models()

    st.subheader("Upload Gambar Air")
    st.caption("Seret & lepas (drag and drop) gambar ke area di bawah, atau klik untuk memilih file.")
    uploaded_file = st.file_uploader(
        "Pilih gambar (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        st.subheader("Preview Gambar")
        st.image(pil_img, caption="Gambar yang diunggah", use_container_width=True)

        with st.spinner("Sedang memproses gambar..."):
            hasil = predict_image(pil_img, rf_model, feature_extractor)

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        if hasil["label"] == "Jernih":
            st.success(f"Jenis Air : **{hasil['label']}**")
        else:
            st.info(f"Jenis Air : **{hasil['label']}**")

        st.write(f"Persentase Kekeruhan : **{hasil['persentase_kekeruhan']}%**")

        st.markdown("### Confidence per Kelas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jernih", f"{hasil['confidence']['Jernih']}%")
        with col2:
            st.metric("Keruh", f"{hasil['confidence']['Keruh']}%")

        st.markdown("### 📊 Visualisasi Probabilitas")
        prob_df = pd.DataFrame(
            {"Probabilitas (%)": hasil["confidence"]},
        )
        st.bar_chart(prob_df, color="#2196F3", height=300)

    else:
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align:center; padding:30px; color:#90A4AE;">
                <h1 style="font-size:48px; margin-bottom:0;">🖼️</h1>
                <p>Belum ada gambar diunggah.<br>
                Silakan unggah gambar air untuk melihat hasil klasifikasi.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📖 Tentang Model CNN + Random Forest"):
            st.write(
                "Model CNN pada aplikasi ini berfungsi sebagai **ekstraktor fitur** "
                "(mengambil pola visual dari gambar melalui layer Flatten), sementara "
                "**Random Forest** bertugas melakukan klasifikasi akhir berdasarkan "
                "fitur tersebut. Kombinasi ini disebut model *hybrid* CNN + Random Forest."
            )

    with st.expander("📖 Informasi Kekeruhan Air"):
        st.markdown(
            """
### 💧 Pengertian Kekeruhan Air

Kekeruhan (*turbidity*) adalah karakteristik optik air yang timbul akibat keberadaan
partikel tersuspensi, baik organik maupun anorganik, yang menyebabkan cahaya yang
melewati air dihamburkan atau diserap alih-alih diteruskan secara lurus. Semakin
tinggi konsentrasi partikel tersuspensi, semakin besar pula intensitas hamburan
cahaya yang terjadi, sehingga air tampak semakin keruh (Nie et al., 2025).

Karena sifatnya yang optik, tingkat kekeruhan tidak hanya dipengaruhi oleh jumlah
partikel, tetapi juga oleh ukuran, bentuk, dan indeks bias partikel tersebut. Hal
ini menjadikan kekeruhan sebagai salah satu indikator penting dalam pemantauan
kualitas air, baik untuk keperluan lingkungan maupun air minum.

### 💙 Pengertian Air Jernih

Air jernih dicirikan oleh tingkat kekeruhan yang rendah akibat sedikitnya partikel
tersuspensi di dalamnya, sehingga cahaya dapat menembus badan air dengan baik.
Kondisi ini umumnya mengindikasikan kualitas air yang relatif baik dan minim
gangguan dari sedimen maupun pencemar (Nie et al., 2025).

Penetrasi cahaya yang baik pada air jernih juga berperan penting bagi ekosistem
perairan, karena mendukung proses fotosintesis organisme akuatik dan menjaga
keseimbangan biologis di dalam air.

### 🤎 Pengertian Air Keruh

Air keruh terbentuk ketika konsentrasi partikel tersuspensi—baik berupa sedimen,
bahan organik, maupun mikroorganisme—meningkat secara signifikan, sehingga
mengurangi kejernihan dan kemampuan cahaya untuk menembus badan air (Nie et al.,
2025). Tingginya tingkat kekeruhan sering kali berkaitan pula dengan meningkatnya
konsentrasi mikroorganisme dan zat pencemar dalam air.

Kondisi air keruh perlu diwaspadai karena dapat menurunkan kualitas air baku,
menghambat proses pengolahan air, serta berpotensi menimbulkan risiko terhadap
kesehatan apabila dikonsumsi tanpa pengolahan yang memadai.

### ⚠️ Faktor Penyebab Air Menjadi Keruh

Kekeruhan air dapat disebabkan oleh berbagai faktor, di antaranya lumpur, sedimen,
dan pasir halus yang terbawa melalui erosi maupun aliran permukaan. Bahan organik
dari peluruhan tumbuhan, alga, serta mikroorganisme turut berkontribusi terhadap
peningkatan partikel tersuspensi dalam badan air (Freshwater Biology – Wiley,
2024).

Selain faktor alami, pertumbuhan alga yang berlebihan (*blooming*) akibat
tingginya kandungan nutrien seperti nitrogen dan fosfor turut meningkatkan
kekeruhan, baik secara langsung melalui sel alga yang tersuspensi maupun secara
tidak langsung lewat interaksi dengan sedimen dasar perairan (KMAE, 2023).
Aktivitas manusia seperti pembuangan limbah dan gangguan terhadap sedimen dasar
juga dapat memperparah kondisi ini (JMSE – MDPI, 2024).

### 📏 Nephelometric Turbidity Unit (NTU)

*Nephelometric Turbidity Unit* (NTU) merupakan satuan standar internasional yang
digunakan untuk menyatakan tingkat kekeruhan air, diperoleh melalui metode
nefelometri yang mengukur intensitas cahaya yang dihamburkan oleh partikel dalam
sampel air pada sudut tertentu terhadap sumber cahaya (ISO 7027-1, dikonfirmasi
2021).

Secara umum, semakin tinggi nilai NTU suatu sampel air, semakin besar tingkat
kekeruhannya dan semakin rendah tingkat kejernihannya. Nilai NTU inilah yang
menjadi acuan kuantitatif dalam berbagai penelitian klasifikasi kekeruhan
berbasis citra digital, termasuk sebagai label data pada model pembelajaran
mesin (Nie et al., 2025).

### 🤖 Cara Kerja Aplikasi

Aplikasi ini melakukan klasifikasi tingkat kekeruhan air berdasarkan
karakteristik visual citra menggunakan pendekatan *hybrid*, yaitu *Convolutional
Neural Network* (CNN) sebagai ekstraktor fitur dan *Random Forest* sebagai
pengklasifikasi akhir. Pendekatan serupa telah digunakan dalam berbagai
penelitian, di mana CNN mampu mempelajari pola visual seperti warna dan tekstur
dari citra air, sementara model *machine learning* konvensional seperti Random
Forest memanfaatkan fitur tersebut untuk melakukan klasifikasi (Sensors – MDPI,
2025; BDCC – MDPI, 2024).

Pendekatan berbasis citra semacam ini telah dikembangkan pula pada penelitian
lain yang menggunakan kamera *smartphone* untuk mengestimasi kekeruhan dan
padatan tersuspensi tanpa memerlukan instrumen turbidimeter (Lopez-Betancur et
al., 2022; Feizi et al., 2022). Perlu dipahami bahwa hasil klasifikasi pada
aplikasi ini merupakan **estimasi berdasarkan citra digital**, bukan pengukuran
langsung menggunakan alat turbidimeter, sehingga akurasinya dapat dipengaruhi
oleh kondisi pencahayaan, kualitas kamera, dan karakteristik sampel air saat
pengambilan gambar (Nie et al., 2025).
            """
        )

    st.markdown("---")
    st.caption(
        "Model: CNN (TensorFlow/Keras) sebagai feature extractor + "
        "Random Forest (scikit-learn) sebagai classifier | "
        "Penulisan Ilmiah - Universitas Gunadarma"
    )


if __name__ == "__main__":
    main()