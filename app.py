"""
Streamlit app - Klasifikasi Cuaca Kolkata (XGBoost)
Jalankan: streamlit run app.py

File yang dipakai / dibuat (permanen di folder yang sama):
- xgboost_weather_artifacts.pkl  : model + imputer + label_encoder (dari notebook / hasil training ulang)
- model_metrics.json             : metrik evaluasi terakhir
- predictions_history.csv        : riwayat semua prediksi (tidak hilang saat refresh/restart)
- kolkata_weather_jan_june_2026.csv : data mentah (kolom 'time' wajib ada) untuk training ulang
"""
import json
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
ARTIFACT_PATH = BASE_DIR / "xgboost_weather_artifacts.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
HISTORY_PATH = BASE_DIR / "predictions_history.csv"
DATA_PATH = BASE_DIR / "kolkata_weather_jan_june_2026.csv"
RANDOM_STATE = 42

FEATURE_COLS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "sunshine_duration", "month", "day_of_year", "day_of_week",
]

WEATHER_LABELS = {
    "0": ("☀️", "Cerah"),
    "1": ("🌤️", "Sebagian besar cerah"),
    "2": ("⛅", "Berawan sebagian"),
    "3": ("☁️", "Mendung"),
    "51": ("🌦️", "Gerimis ringan"),
    "53": ("🌦️", "Gerimis sedang"),
    "55": ("🌧️", "Gerimis lebat"),
    "61": ("🌧️", "Hujan ringan"),
    "63": ("🌧️", "Hujan sedang"),
    "65": ("⛈️", "Hujan lebat"),
}


def label_of(code: str) -> str:
    icon, name = WEATHER_LABELS.get(str(code), ("❔", "Tidak diketahui"))
    return f"{icon} {name} (kode {code})"


# ----------------------------------------------------------------------------
# Model: muat / latih / simpan
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    if ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "rb") as f:
            return pickle.load(f)
    return None


def build_training_frame(df: pd.DataFrame):
    if "time" not in df.columns or "weather_code" not in df.columns:
        raise ValueError(
            "CSV harus berisi kolom mentah 'time' dan 'weather_code'. "
            "Catatan: file CSV hasil normalisasi di notebook tidak punya kolom 'time'."
        )
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data["month"] = data["time"].dt.month
    data["day_of_year"] = data["time"].dt.dayofyear
    data["day_of_week"] = data["time"].dt.dayofweek
    data = data.dropna(subset=["time", "weather_code"]).reset_index(drop=True)
    return data[FEATURE_COLS], data["weather_code"].astype(int).astype(str)


def train_and_save(df: pd.DataFrame, n_estimators=100, learning_rate=0.1, max_depth=5, test_size=0.2):
    X, y_raw = build_training_frame(df)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_tr_i = imputer.fit_transform(X_tr)
    X_te_i = imputer.transform(X_te)

    model = XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        objective="multi:softprob", num_class=len(le.classes_),
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_tr_i, y_tr)
    y_pred = model.predict(X_te_i)

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision_macro": float(precision_score(y_te, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "classes": list(le.classes_),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=np.arange(len(le.classes_))).tolist(),
        "feature_importance": dict(zip(FEATURE_COLS, map(float, model.feature_importances_))),
    }
    artifacts = {
        "model": model, "imputer": imputer, "label_encoder": le,
        "feature_cols": FEATURE_COLS, "class_names": list(le.classes_),
    }
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    load_artifacts.clear()  # paksa cache dimuat ulang
    return metrics


def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


# ----------------------------------------------------------------------------
# Riwayat prediksi (permanen ke CSV)
# ----------------------------------------------------------------------------
def append_history(rows: pd.DataFrame):
    rows = rows.copy()
    rows.insert(0, "waktu_prediksi", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if HISTORY_PATH.exists():
        old = pd.read_csv(HISTORY_PATH, dtype={"kode_cuaca": str})
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(HISTORY_PATH, index=False)


def load_history():
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH, dtype={"kode_cuaca": str})
    return pd.DataFrame()


def predict(artifacts, X: pd.DataFrame):
    Xi = artifacts["imputer"].transform(X[artifacts["feature_cols"]])
    proba = artifacts["model"].predict_proba(Xi)
    idx = proba.argmax(axis=1)
    codes = artifacts["label_encoder"].inverse_transform(idx)
    return codes, proba


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Cuaca Kolkata", page_icon="🌦️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .hero {background: linear-gradient(120deg,#1e3c72,#2a5298); color:#fff;
           padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size: 1.9rem;}
    .hero p {margin:.3rem 0 0; opacity:.9;}
    .result {background:#f0f6ff; border-left: 6px solid #2a5298; padding:1rem 1.2rem;
             border-radius:10px; font-size:1.25rem; color:#12294d;}
    </style>
    <div class="hero">
      <h1>🌦️ Klasifikasi Cuaca Kolkata</h1>
      <p>Prediksi kode cuaca harian (WMO) menggunakan XGBoost · Aldi Eka Saputra</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifacts = load_artifacts()

# Sidebar
with st.sidebar:
    st.header("⚙️ Status")
    if artifacts:
        st.success("Model tersimpan & dimuat")
        m = load_metrics()
        if m:
            st.caption(f"Dilatih: {m['trained_at']}")
            st.metric("Accuracy", f"{m['accuracy']:.2%}")
            st.metric("F1 (macro)", f"{m['f1_macro']:.2%}")
    else:
        st.warning("Model belum ada. Buka tab **Model** untuk melatih dari CSV.")
    hist = load_history()
    st.metric("Total prediksi tersimpan", len(hist))
    st.caption("Semua hasil tersimpan permanen di file lokal (pkl/json/csv).")

tab_pred, tab_batch, tab_hist, tab_model = st.tabs(
    ["🔮 Prediksi", "📄 Prediksi Batch (CSV)", "🗂️ Riwayat", "🧠 Model & Evaluasi"]
)

# --- Tab Prediksi tunggal -----------------------------------------------------
with tab_pred:
    if not artifacts:
        st.info("Latih atau letakkan `xgboost_weather_artifacts.pkl` terlebih dahulu.")
    else:
        with st.form("form_pred"):
            c0, _ = st.columns([1, 2])
            tgl = c0.date_input("Tanggal", value=date.today())
            c1, c2, c3 = st.columns(3)
            t_mean = c1.number_input("Suhu rata-rata (°C)", value=27.0, step=0.1)
            t_max = c2.number_input("Suhu maks (°C)", value=31.0, step=0.1)
            t_min = c3.number_input("Suhu min (°C)", value=23.0, step=0.1)
            c4, c5, c6 = st.columns(3)
            precip = c4.number_input("Total presipitasi (mm)", value=0.0, min_value=0.0, step=0.1)
            rain = c5.number_input("Total hujan (mm)", value=0.0, min_value=0.0, step=0.1)
            p_hours = c6.number_input("Jam presipitasi", value=0.0, min_value=0.0, max_value=24.0, step=1.0)
            c7, c8, c9 = st.columns(3)
            wind = c7.number_input("Kecepatan angin maks (km/j)", value=13.0, min_value=0.0, step=0.1)
            gust = c8.number_input("Hembusan angin maks (km/j)", value=30.0, min_value=0.0, step=0.1)
            wdir = c9.number_input("Arah angin dominan (°)", value=190, min_value=0, max_value=360, step=1)
            sun = st.slider("Durasi sinar matahari (detik)", 0, 45000, 33000, step=100)
            submit = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

        if submit:
            row = pd.DataFrame([{
                "temperature_2m_mean": t_mean, "temperature_2m_max": t_max, "temperature_2m_min": t_min,
                "precipitation_sum": precip, "rain_sum": rain, "precipitation_hours": p_hours,
                "wind_speed_10m_max": wind, "wind_gusts_10m_max": gust,
                "wind_direction_10m_dominant": wdir, "sunshine_duration": sun,
                "month": tgl.month, "day_of_year": tgl.timetuple().tm_yday, "day_of_week": tgl.weekday(),
            }])
            codes, proba = predict(artifacts, row)
            code = str(codes[0])
            conf = float(proba[0].max())
            st.markdown(
                f'<div class="result"><b>Hasil:</b> {label_of(code)}<br>Keyakinan: <b>{conf:.1%}</b></div>',
                unsafe_allow_html=True,
            )
            prob_df = pd.DataFrame({
                "Kode": artifacts["class_names"],
                "Cuaca": [label_of(c) for c in artifacts["class_names"]],
                "Probabilitas": proba[0],
            }).sort_values("Probabilitas", ascending=False)
            fig = px.bar(prob_df, x="Probabilitas", y="Cuaca", orientation="h",
                         title="Probabilitas tiap kelas")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
            st.plotly_chart(fig, use_container_width=True)

            save = row.copy()
            save.insert(0, "tanggal", tgl.isoformat())
            save["kode_cuaca"] = code
            save["keterangan"] = WEATHER_LABELS.get(code, ("", "-"))[1]
            save["keyakinan"] = round(conf, 4)
            append_history(save)
            st.success("Hasil prediksi disimpan permanen ke riwayat ✅")

# --- Tab Batch ----------------------------------------------------------------
with tab_batch:
    if not artifacts:
        st.info("Model belum tersedia.")
    else:
        st.write("Unggah CSV dengan kolom fitur mentah (tidak dinormalisasi). "
                 "Jika ada kolom `time`, kolom month/day_of_year/day_of_week dibuat otomatis.")
        st.code(", ".join(FEATURE_COLS), language="text")
        up = st.file_uploader("File CSV", type="csv")
        if up is not None:
            df_up = pd.read_csv(up)
            if "time" in df_up.columns:
                t = pd.to_datetime(df_up["time"], errors="coerce")
                df_up["month"], df_up["day_of_year"], df_up["day_of_week"] = t.dt.month, t.dt.dayofyear, t.dt.dayofweek
            missing = [c for c in FEATURE_COLS if c not in df_up.columns]
            if missing:
                st.error(f"Kolom belum lengkap: {missing}")
            else:
                codes, proba = predict(artifacts, df_up)
                out = df_up.copy()
                out["kode_cuaca"] = codes.astype(str)
                out["keterangan"] = [WEATHER_LABELS.get(c, ("", "-"))[1] for c in out["kode_cuaca"]]
                out["keyakinan"] = proba.max(axis=1).round(4)
                st.dataframe(out, use_container_width=True)
                st.plotly_chart(
                    px.histogram(out, x="keterangan", title="Distribusi hasil prediksi"),
                    use_container_width=True,
                )
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ Unduh hasil", out.to_csv(index=False).encode(),
                                   "hasil_prediksi.csv", "text/csv")
                if c2.button("💾 Simpan ke riwayat permanen"):
                    to_save = out.copy()
                    to_save.insert(0, "tanggal", to_save["time"].astype(str) if "time" in to_save else "")
                    append_history(to_save[["tanggal"] + FEATURE_COLS + ["kode_cuaca", "keterangan", "keyakinan"]])
                    st.success("Tersimpan ✅")

# --- Tab Riwayat --------------------------------------------------------------
with tab_hist:
    hist = load_history()
    if hist.empty:
        st.info("Belum ada riwayat prediksi.")
    else:
        st.dataframe(hist.iloc[::-1], use_container_width=True)
        st.plotly_chart(px.histogram(hist, x="keterangan", title="Distribusi prediksi tersimpan"),
                        use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Unduh riwayat", hist.to_csv(index=False).encode(),
                           "predictions_history.csv", "text/csv")
        with c2.popover("🗑️ Hapus semua riwayat"):
            st.warning("Tindakan ini tidak bisa dibatalkan.")
            if st.button("Ya, hapus permanen"):
                HISTORY_PATH.unlink(missing_ok=True)
                st.rerun()

# --- Tab Model ----------------------------------------------------------------
with tab_model:
    m = load_metrics()
    if m:
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m['accuracy']:.2%}")
        c[1].metric("Precision (macro)", f"{m['precision_macro']:.2%}")
        c[2].metric("Recall (macro)", f"{m['recall_macro']:.2%}")
        c[3].metric("F1 (macro)", f"{m['f1_macro']:.2%}")
        fi = pd.Series(m["feature_importance"]).sort_values().rename("importance").reset_index()
        fi.columns = ["fitur", "importance"]
        st.plotly_chart(px.bar(fi, x="importance", y="fitur", orientation="h",
                               title="Feature Importance"), use_container_width=True)
        cm = np.array(m["confusion_matrix"])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=m["classes"], y=m["classes"],
                                  labels=dict(x="Prediksi", y="Aktual"),
                                  color_continuous_scale="Blues", title="Confusion Matrix"),
                        use_container_width=True)
    else:
        st.info("Belum ada metrik tersimpan (hanya muncul setelah training dari aplikasi ini).")

    st.subheader("Latih ulang model")
    st.caption("Menggunakan CSV data mentah (harus ada kolom `time` & `weather_code`). "
               "Hasil model & metrik disimpan permanen.")
    src = st.file_uploader("CSV data mentah (opsional, default: file di folder aplikasi)", type="csv", key="train_csv")
    p1, p2, p3 = st.columns(3)
    n_est = p1.slider("n_estimators", 50, 500, 100, 10)
    lr = p2.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3], value=0.1)
    depth = p3.slider("max_depth", 2, 10, 5)
    if st.button("🚀 Latih & simpan model", type="primary"):
        try:
            if src is not None:
                df_train = pd.read_csv(src)
            elif DATA_PATH.exists():
                df_train = pd.read_csv(DATA_PATH)
            else:
                raise FileNotFoundError("Tidak ada data. Unggah CSV mentah.")
            with st.spinner("Melatih model..."):
                res = train_and_save(df_train, n_est, lr, depth)
            st.success(f"Selesai. Accuracy {res['accuracy']:.2%} — model tersimpan permanen.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
