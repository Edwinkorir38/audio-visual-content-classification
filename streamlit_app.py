# streamlit_app.py — Full merged app with advanced theme (top + sidebar toggles)
# Features:
# - warnings suppressed (InconsistentVersionWarning)
# - cached model/component loading
# - NLTK ensure + caching
# - cleaned preprocessing
# - model confidence with probabilities & bar chart
# - CAI visualization
# - simple feature-importance (if available)
# - logging predictions to CSV + download
# - prettier UI with sidebar and tabs
# - advanced theme selector + synchronized top & sidebar dark toggles

import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ---------------------- Silence specific warnings -------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- Imports -------------------------
import os
from datetime import datetime, date
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# ---------------------- App config (must be before other st.* usage) -------------------------
st.set_page_config(
    page_title="🎬 Film Rating Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Constants -------------------------
LOG_CSV = "prediction_log.csv"

# ---------------------- Ensure log exists -------------------------
def ensure_log_exists():
    if not os.path.exists(LOG_CSV):
        df_log = pd.DataFrame(
            columns=[
                "timestamp", "duration_mins", "classification_date",
                "genre", "country_of_origin", "selected_advisories",
                "synopsis", "justification", "predicted_rating"
            ]
        )
        df_log.to_csv(LOG_CSV, index=False)

ensure_log_exists()

# ---------------------- 1) NLTK data -------------------------
@st.cache_resource
def ensure_nltk_data():
    try:
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/omw-1.4")
        return True
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        return True

ensure_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.update([
    'film','movie','story','character','new','one','two','three','etc',
    'no synopsis available','no justification given','missing synopsis','unknown',
    'justification','reason','reasons','scene','scenes','visual','language','content',
    'implied','strong','suggestive','brief','sequences','thematic','elements','some',
    'mild','minor','depictions','references','dialogue','material','moderate','explicit',
    'disturbing','images','action','fantasy','horror','peril','sexual','violence',
    'drug','abuse','children','adult','words','word','rated','rating'
])

# ---------------------- 2) Load components (cached) -------------------------
@st.cache_resource
def load_components():
    required_files = [
        "best_model.pkl",
        "fitted_preprocessor.pkl",
        "tfidf_synopsis.pkl",
        "tfidf_justification.pkl",
        "categorical_ohe_encoder.pkl",
        "rating_order.pkl",
        "X_columns.pkl",
        "X_raw_columns.pkl",
    ]

    for fn in required_files:
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Required file missing: {fn}")

    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("fitted_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("tfidf_synopsis.pkl", "rb") as f:
        tfidf_synopsis = pickle.load(f)

    with open("tfidf_justification.pkl", "rb") as f:
        tfidf_justification = pickle.load(f)

    with open("categorical_ohe_encoder.pkl", "rb") as f:
        categorical_ohe_encoder = pickle.load(f)

    with open("rating_order.pkl", "rb") as f:
        rating_order = pickle.load(f)

    with open("X_columns.pkl", "rb") as f:
        X_columns_transformed = pickle.load(f)

    with open("X_raw_columns.pkl", "rb") as f:
        X_columns_raw = pickle.load(f)

    return (
        tfidf_synopsis,
        tfidf_justification,
        categorical_ohe_encoder,
        preprocessor,
        rating_order,
        X_columns_transformed,
        X_columns_raw,
        model,
    )

try:
    (
        tfidf_synopsis,
        tfidf_justification,
        categorical_ohe_encoder,
        preprocessor,
        rating_order,
        X_columns_transformed,
        X_columns_raw,
        model,
    ) = load_components()
except Exception as e:
    st.error("Failed to load ML components: " + str(e))
    st.stop()

# ---------------------- Helpers -------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def safe_predict_proba(model, X):
    try:
        probs = model.predict_proba(X)
        return probs
    except Exception:
        try:
            scores = model.decision_function(X)
            if scores.ndim == 1:
                from scipy.special import expit
                p1 = expit(scores)
                probs = np.vstack([1 - p1, p1]).T
            else:
                e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)
            return probs
        except Exception:
            return None

def get_feature_importances(model, X_columns):
    try:
        est = model
        if hasattr(model, "steps"):
            est = model.steps[-1][1]

        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
            fi = pd.Series(importances, index=X_columns).sort_values(ascending=False)
            return fi.head(20)
        if hasattr(est, "coef_"):
            coef = est.coef_
            if coef.ndim > 1:
                coef = np.sum(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            fi = pd.Series(coef, index=X_columns).sort_values(ascending=False)
            return fi.head(20)
    except Exception:
        return None

def log_prediction(row_dict: dict):
    df = pd.DataFrame([row_dict])
    df.to_csv(LOG_CSV, mode="a", header=False, index=False)

# -------------------------------
# Theme: Advanced + synchronized top & sidebar toggles
# -------------------------------
# initialize session_state for dark_mode if missing
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Sidebar theme checkbox
with st.sidebar:
    st.markdown("## 🌓 Theme")
    sidebar_dark = st.checkbox("Dark Mode (sidebar)", value=st.session_state["dark_mode"], key="sidebar_dark_mode")

# Top theme checkbox (rendered in header area)
# We'll show it just below the page header so user can toggle there too
top_dark = st.checkbox("Dark Mode (top)", value=st.session_state["dark_mode"], key="top_dark_mode")

# Sync toggles: if either changed, set session_state["dark_mode"] accordingly
# (Note: Streamlit runs the script top-to-bottom each interaction; this keeps both checkboxes consistent.)
if sidebar_dark != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = sidebar_dark
if top_dark != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = top_dark

dark_mode = st.session_state["dark_mode"]

def apply_advanced_theme(dark: bool):
    if dark:
        st.markdown(
            """
            <style>
            :root {
                --bg-color: #0e0e0e;
                --card-color: #1c1c1c;
                --text-color: #e6e6e6;
                --accent: #00d4ff;
                --input-bg: #2b2b2b;
                --input-border: #00d4ffaa;
            }

            body, .stApp {
                background-color: var(--bg-color) !important;
                color: var(--text-color) !important;
                transition: background-color 0.4s ease, color 0.4s ease;
            }

            h1, h2, h3, h4 {
                color: var(--accent) !important;
            }

            section[data-testid="stSidebar"] {
                background-color: var(--card-color) !important;
                border-right: 1px solid #333 !important;
            }

            .stTextInput, .stSelectbox, .stNumberInput, .stTextArea {
                background-color: var(--card-color) !important;
            }

            input, textarea, select {
                background-color: var(--input-bg) !important;
                color: var(--text-color) !important;
                border: 1px solid var(--input-border) !important;
                border-radius: 6px !important;
                transition: border 0.3s ease;
            }

            input:focus, textarea:focus, select:focus {
                border: 1px solid var(--accent) !important;
                box-shadow: 0 0 8px var(--accent) !important;
            }

            .stButton>button {
                background-color: var(--accent) !important;
                color: #000 !important;
                border-radius: 8px !important;
                border: none !important;
                font-weight: 600 !important;
                transition: 0.25s ease-in-out;
            }

            .stButton>button:hover {
                background-color: #00eaff !important;
                transform: scale(1.03);
                box-shadow: 0 0 12px var(--accent) !important;
            }

            .dataframe, .stJson {
                color: var(--text-color) !important;
                background-color: var(--card-color) !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            :root {
                --accent: #007bff;
            }

            h1, h2, h3 {
                color: var(--accent) !important;
            }

            .stButton>button {
                background-color: var(--accent) !important;
                color: white !important;
                border-radius: 6px !important;
                transition: 0.25s ease-in-out;
            }

            .stButton>button:hover {
                background-color: #3399ff !important;
                transform: scale(1.03);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

apply_advanced_theme(dark_mode)

# ---------------------- Top floating header (title) -------------------------
st.markdown(
    """
    <style>
    .top-bar {
        position: sticky;
        top: 0;
        z-index: 999;
        padding: 10px 20px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 8px;
    }
    .top-bar-title {
        font-size: 24px;
        font-weight: 700;
    }
    </style>
    <div class="top-bar"><span class="top-bar-title">🎬 Film Rating Predictor</span></div>
    """,
    unsafe_allow_html=True
)

# ---------------------- Main layout: tabs -------------------------
tab1, tab2, tab3 = st.tabs(["Predict", "Visualize & Explain", "About & Deploy"])

# ---------- Tab 1: Predict ----------
with tab1:
    st.subheader("Enter film details")
    with st.form("film_input_form", clear_on_submit=False):
        # Expanders for nicer UI
        with st.expander("🎛️ Film Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                duration_mins = st.number_input("Duration (minutes)", min_value=1, max_value=500, value=90)
                genre_categories = categorical_ohe_encoder.categories_[0].tolist()
                genre = st.selectbox("Genre", options=genre_categories, index=genre_categories.index('drama') if 'drama' in genre_categories else 0)
            with col2:
                country_categories = categorical_ohe_encoder.categories_[1].tolist()
                country_of_origin = st.selectbox("Country", options=country_categories, index=country_categories.index('United States') if 'United States' in country_categories else 0)
                classification_date = st.date_input("Classification Date", value=date.today())
                classification_year = classification_date.year
                classification_month = classification_date.month
                classification_day_of_week = classification_date.weekday()
                title = st.text_input("Title (optional)", "")

        with st.expander("📝 Synopsis & Justification", expanded=True):
            synopsis = st.text_area("Synopsis", value="A compelling story about a group of friends.")
            justification = st.text_area("Justification (optional)", value="Contains mild themes and coarse language.")

        advisory_categories = sorted(list(set([
            'Violence','Language','Sex','Horror','Alcohol','Crime','Drugs','Nudity',
            'Occultism','Other','Parental Guidance','Profanity','Theme','Betting',
            'Kissing','Fright','Mature','Harmful Imitable','Restricted','Suicide',
            'Coarse Language','Obscenity','Horror/Scary','Weapon','Historical',
            'Medical','Community','Gambling','LGBTQ','Content','UNSPECIFIED'
        ])))

        with st.expander("⚠️ Consumer Advisory", expanded=False):
            selected_advisories = st.multiselect("Consumer Advisory Index (select all that apply)", options=advisory_categories, default=['Violence','Language'] if 'Violence' in advisory_categories else [])

        colA, colB = st.columns([1,1])
        with colA:
            submitted = st.form_submit_button("🔮 Predict Rating Now")
        with colB:
            if st.form_submit_button("🧹 Clear Inputs"):
                # Clear - basic approach (re-run will reset form defaults next time)
                st.experimental_rerun()

    if submitted:
        with st.spinner("Preprocessing and predicting..."):
            # Build input df
            input_df = pd.DataFrame([{
                'duration_mins': duration_mins,
                'classification_year': classification_year,
                'classification_month': classification_month,
                'classification_day_of_week': classification_day_of_week,
                'synopsis': synopsis,
                'justification': justification,
                'genre': genre,
                'country_of_origin': country_of_origin
            }])

            # Text features
            input_df["synopsis_processed"] = input_df["synopsis"].apply(preprocess_text)
            input_df["justification_processed"] = input_df["justification"].apply(preprocess_text)

            syn_tfidf = tfidf_synopsis.transform(input_df["synopsis_processed"])
            jus_tfidf = tfidf_justification.transform(input_df["justification_processed"])

            df_syn = pd.DataFrame(syn_tfidf.toarray(), columns=[f"synopsis_{c}" for c in tfidf_synopsis.get_feature_names_out()])
            df_jus = pd.DataFrame(jus_tfidf.toarray(), columns=[f"justification_{c}" for c in tfidf_justification.get_feature_names_out()])

            # OHE
            df_ohe = pd.DataFrame(categorical_ohe_encoder.transform(input_df[['genre','country_of_origin']]).toarray(),
                                  columns=categorical_ohe_encoder.get_feature_names_out(['genre','country_of_origin']))

            # CAI
            df_cai = pd.DataFrame({f"CAI_{adv}": [1 if adv in selected_advisories else 0] for adv in advisory_categories})

            # Concat everything
            X_temp = pd.concat([
                input_df[['duration_mins','classification_year','classification_month','classification_day_of_week']],
                df_syn, df_jus, df_ohe, df_cai
            ], axis=1)

            # Align with training cols
            final_df = pd.DataFrame(0, index=[0], columns=X_columns_raw)
            for col in X_temp.columns:
                if col in final_df.columns:
                    final_df[col] = X_temp[col]

            # Prediction
            try:
                pred_idx = model.predict(final_df)[0]
            except Exception as e:
                st.error("Model failed to predict: " + str(e))
                pred_idx = None

            predicted_rating = rating_order[pred_idx] if pred_idx is not None else "ERROR"

            # Try probabilities
            probs = safe_predict_proba(model, final_df)
            prob_series = None
            if probs is not None:
                classes = getattr(model, "classes_", None)
                if classes is None:
                    try:
                        if hasattr(model, "steps"):
                            classes = model.steps[-1][1].classes_
                    except Exception:
                        classes = None
                if classes is None:
                    classes = list(rating_order.keys())
                # Map numeric class indices to readable labels via rating_order
                # rating_order is expected to be dict-like mapping index->label
                try:
                    prob_indexed = [rating_order[c] if c in rating_order else str(c) for c in classes]
                except Exception:
                    prob_indexed = [str(c) for c in classes]
                prob_series = pd.Series(probs[0], index=prob_indexed)
                prob_series.index = [str(i) for i in prob_series.index]

            # Show results
            st.success(f"Predicted Classification: **{predicted_rating}**")
            if prob_series is not None:
                st.subheader("Model confidence (probabilities)")
                prob_df_display = (prob_series * 100).round(2).rename("percent")
                st.dataframe(prob_df_display.to_frame())

                # Bar chart
                st.bar_chart(prob_df_display)
            else:
                st.info("Model does not provide probabilities; confidence chart not available.")

            # show raw input
            st.markdown("**Input used for prediction:**")
            st.json({
                "title": title,
                "duration_mins": duration_mins,
                "classification_date": classification_date.isoformat(),
                "genre": genre,
                "country_of_origin": country_of_origin,
                "selected_advisories": selected_advisories,
                "synopsis": synopsis,
                "justification": justification
            })

            # Save log row
            log_row = {
                "timestamp": datetime.utcnow().isoformat(),
                "duration_mins": duration_mins,
                "classification_date": classification_date.isoformat(),
                "genre": genre,
                "country_of_origin": country_of_origin,
                "selected_advisories": "|".join(selected_advisories),
                "synopsis": synopsis,
                "justification": justification,
                "predicted_rating": predicted_rating
            }
            try:
                log_prediction(log_row)
                st.caption("Prediction logged locally.")
            except Exception:
                st.caption("Warning: failed to write to local log file (permission issue?).")

# ---------- Tab 2: Visualize & Explain ----------
with tab2:
    st.subheader("Visualizations & Explanation")

    # Show last N log entries chart
    if os.path.exists(LOG_CSV):
        df_log = pd.read_csv(LOG_CSV)
        if not df_log.empty:
            st.markdown("**Recent predictions**")
            st.dataframe(df_log.tail(20))

            # Distribution of predicted ratings
            rating_counts = df_log['predicted_rating'].value_counts().sort_index()
            st.markdown("**Prediction distribution**")
            st.bar_chart(rating_counts)

    # CAI preview
    st.markdown("**Consumer Advisory (CAI) indicator preview**")
    try:
        cai_preview = pd.Series({f"CAI_{adv}": (1 if adv in selected_advisories else 0) for adv in advisory_categories})
    except Exception:
        cai_preview = pd.Series()
    if not cai_preview.empty and cai_preview.sum() > 0:
        st.write("Selected advisories highlighted (1 = selected)")
        st.dataframe(cai_preview.to_frame("selected").sort_values("selected", ascending=False))
        fig, ax = plt.subplots(figsize=(6, max(2, cai_preview.sum() * 0.25 + 1)))
        selected = cai_preview[cai_preview == 1]
        if not selected.empty:
            selected.sort_values().plot(kind='barh', ax=ax)
            ax.set_xlabel("Selected (1)")
            ax.set_ylabel("Advisory")
            st.pyplot(fig)
        else:
            st.info("No advisories selected in current session.")
    else:
        st.info("No CAI data available yet — run a prediction first and select advisories.")

    # Feature importance (best-effort)
    st.markdown("---")
    st.subheader("Simple feature importance (best-effort)")
    fi = get_feature_importances(model, X_columns_raw)
    if fi is not None:
        st.write("Top features used by model (approximate).")
        st.dataframe(fi.to_frame("importance"))
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fi.head(10).sort_values().plot(kind="barh", ax=ax2)
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)
    else:
        st.info("Model does not expose feature_importances_ or coef_; feature importances not available.")

    # Download log
    if os.path.exists(LOG_CSV):
        st.markdown("---")
        st.download_button("Download prediction log (CSV)", data=open(LOG_CSV, "rb"), file_name="prediction_log.csv")

# ---------- Tab 3: About & Deploy ----------
with tab3:
    st.header("About this app")
    st.markdown(
        """
This app predicts film classification ratings (e.g., GE, PG, 16, 18, R) using a pre-trained ML pipeline.
It includes:
- TF-IDF text features for synopsis & justification
- One-hot encoding for genre & country
- Consumer Advisory (CAI) dummy variables
- A Scikit-learn pipeline (loaded from best_model.pkl)
"""
    )
    st.markdown("**Deployment (quick guide)**")
    st.markdown(
        """
1. Create a GitHub repository and push this project. Include all `.pkl` files and `requirements.txt`.
2. Sign in to Streamlit Cloud (https://streamlit.io/cloud).
3. Create a new app and point it to the repository + branch.
4. Set the start command: `streamlit run streamlit_app.py`.
5. If needed, pin exact package versions in `requirements.txt`, for example:
* scikit-learn==1.5.2
* pandas
* numpy
* streamlit
* nltk
* matplotlib

6. Deploy — Streamlit Cloud will build automatically.
"""
    )

    st.markdown("Made with ❤️ using Streamlit and Scikit-learn.")

# End of file
