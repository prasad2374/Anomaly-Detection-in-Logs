import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import re

st.set_page_config(page_title="Log/CSV Analyzer with Anomaly Detection", layout="wide")
st.title("📄 Log/CSV TF-IDF Analyzer + 🧠 Anomaly Detection")

# ---------- Helper Functions ----------
def clean_log_line(line):
    line = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d+)?", "", line)
    line = re.sub(r"[A-Z][a-z]{2} [ \d]{1,2} \d{2}:\d{2}:\d{2}", "", line)
    line = re.sub(r"\b(INFO|DEBUG|ERROR|WARNING|WARN|CRITICAL|NOTICE|SEVERE)\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\[.*?\]", "", line)
    line = re.sub(r"\b(pid|PID)=\d+", "", line)
    line = re.sub(r"\d+\.\d+\.\d+\.\d+", "", line)
    line = re.sub(r"(/[a-zA-Z0-9_\-./]+)", "", line)
    return re.sub(r"\s+", " ", line).strip()

def run_tfidf(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def detect_anomalies(X, contamination=0.05):
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)
    return preds

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload a `.log` or `.csv` file", type=["log", "csv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "log":
        st.subheader("📝 Processing .log File")
        raw_lines = uploaded_file.read().decode(errors='ignore').splitlines()
        cleaned = [clean_log_line(line) for line in raw_lines if len(line.strip()) > 5]
    else:
        st.subheader("🧾 Processing .csv File")
        df = pd.read_csv(uploaded_file)
        st.write("CSV Preview:", df.head())
        text_column = st.selectbox("Select the column containing log text", df.columns)
        cleaned = df[text_column].fillna("").astype(str).apply(clean_log_line).tolist()

    # ---------- TF-IDF ----------
    X, vectorizer = run_tfidf(cleaned)
    st.success(f"TF-IDF computed for {len(cleaned)} entries")

    # ---------- Anomaly Detection ----------
    if st.checkbox("Run Anomaly Detection"):
        contamination = st.slider("Select anomaly sensitivity (higher = more anomalies)", 0.01, 0.3, 0.05)
        preds = detect_anomalies(X, contamination=contamination)
        df_result = pd.DataFrame({
            "Log Entry": cleaned,
            "Anomaly": preds
        })

        anomalies = df_result[df_result["Anomaly"] == -1]
        st.warning(f"⚠️ Detected {len(anomalies)} anomalies")
        st.dataframe(anomalies)

        if st.checkbox("Show all entries with labels"):
            st.dataframe(df_result)

    # ---------- Top Entries ----------
    st.subheader("🔝 Top Log Entries by TF-IDF")
    scores = X.sum(axis=1)
    top_n = st.slider("Select number of top log messages to show", 10, 100, 25)
    top_indices = scores.argsort(axis=0)[-top_n:][::-1].A1
    top_entries = [cleaned[i] for i in top_indices]
    top_scores = [scores[i, 0] for i in top_indices]
    df_top = pd.DataFrame({"Log Entry": top_entries, "TF-IDF Score": top_scores})
    st.dataframe(df_top)

    # ---------- Keyword Importance ----------
    if st.checkbox("Show top keywords across logs"):
        feature_array = vectorizer.get_feature_names_out()
        tfidf_sorting = X.sum(axis=0).A1.argsort()[::-1]
        keyword_n = st.slider("Number of keywords to display", 5, 50, 15)
        top_keywords = [(feature_array[i], X.sum(axis=0).A1[i]) for i in tfidf_sorting[:keyword_n]]
        keyword_df = pd.DataFrame(top_keywords, columns=["Keyword", "Importance"])
        st.dataframe(keyword_df)
else:
    st.info("Upload a `.log` or `.csv` file to begin analysis.")
