import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import date

# --- NLTK Data Download Function ---
@st.cache_resource # This decorator caches the function's output, so it runs only once.
def ensure_nltk_data():
    """
    Ensures that necessary NLTK data (wordnet, stopwords, omw-1.4) is downloaded.
    Uses st.cache_resource to prevent repeated downloads.
    """
    st.info("Checking and downloading NLTK data (wordnet, stopwords, omw-1.4)... This may take a moment on first run.")
    try:
        # Check if data is already present without downloading
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/omw-1.4')
        st.success("NLTK data already available.")
    except LookupError: # Correct exception for nltk.data.find()
        # If not found, attempt to download
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('omw-1.4', quiet=True) # Open Multilingual WordNet, often needed for WordNetLemmatizer
            st.success("NLTK data downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download NLTK data. Please check your internet connection and permissions. Error: {e}")
            st.stop() # Stop the app if essential data cannot be downloaded
    except Exception as e:
        st.error(f"An unexpected error occurred while checking NLTK data: {e}")
        st.stop()

# Call the function at the start of your script to ensure data is present
ensure_nltk_data()

# Now, initialize your NLTK components (lemmatizer, stop_words) after ensuring data is downloaded
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.update([
    'film', 'movie', 'story', 'character', 'new', 'one', 'two', 'three', 'etc',
    'no synopsis available', 'no justification given', 'missing synopsis', 'unknown',
    'justification', 'reason', 'reasons',
    'scene', 'scenes', 'visual', 'language', 'content', 'implied', 'strong',
    'suggestive', 'brief', 'sequences', 'thematic', 'elements', 'some', 'mild',
    'minor', 'depictions', 'references', 'dialogue', 'material', 'moderate',
    'explicit', 'disturbing', 'images', 'action', 'fantasy', 'horror', 'peril',
    'sexual', 'violence', 'drug', 'abuse', 'children', 'adult', 'words', 'word',
    'rated', 'rating',
])

# Load all necessary pre-trained components
@st.cache_resource # Cache the loading of heavy objects
def load_components():
    try:
        st.info("Loading best_model.pkl...")
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        st.info("Loading fitted_preprocessor.pkl...")
        with open('fitted_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        st.info("Loading tfidf_synopsis.pkl...")
        with open('tfidf_synopsis.pkl', 'rb') as f:
            tfidf_synopsis = pickle.load(f)
        
        st.info("Loading tfidf_justification.pkl...")
        with open('tfidf_justification.pkl', 'rb') as f:
            tfidf_justification = pickle.load(f)
        
        st.info("Loading categorical_ohe_encoder.pkl...")
        with open('categorical_ohe_encoder.pkl', 'rb') as f:
            categorical_ohe_encoder = pickle.load(f)
        
        st.info("Loading rating_order.pkl...")
        with open('rating_order.pkl', 'rb') as f:
            rating_order = pickle.load(f)
        
        st.info("Loading X_columns.pkl (transformed names)...")
        with open('X_columns.pkl', 'rb') as f: # This holds TRANSFORMED column names for the model
            X_columns_transformed = pickle.load(f) # Renamed for clarity

        st.info("Loading X_raw_columns.pkl (raw names for preprocessor input)...")
        with open('X_raw_columns.pkl', 'rb') as f: # This holds RAW column names for preprocessor input
            X_columns_raw = pickle.load(f) # New variable
            
        st.success("All components loaded successfully!")
        return tfidf_synopsis, tfidf_justification, categorical_ohe_encoder, preprocessor, rating_order, X_columns_transformed, X_columns_raw, model # Adjusted return values
    except FileNotFoundError as e:
        st.error(f"Error loading a required file: {e}. Please ensure all .pkl files are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during component loading: {e}")
        st.error("This often means a .pkl file is corrupted or empty. Please re-run your training notebook completely.")
        st.stop()

# Adjust unpacking of returned values
tfidf_synopsis, tfidf_justification, categorical_ohe_encoder, preprocessor, rating_order, X_columns_transformed, X_columns_raw, model = load_components()

# --- Text Preprocessing Function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# --- Streamlit UI ---
st.set_page_config(page_title="Film Rating Predictor", layout="wide")

st.title("🎬 Film Rating Predictor")
st.markdown("""
    Enter the details of a film below to predict its classification rating (GE, PG, 16, 18, R).
    This model was trained on historical film classification data.
""")

# Input fields
with st.form("film_input_form"):
    st.header("Film Details")

    col1, col2 = st.columns(2)
    with col1:
        duration_mins = st.number_input("Duration (minutes)", min_value=1, max_value=500, value=90)
        
        # Get categories from the fitted OHE encoder for dropdowns
        # The OHE encoder was fitted on ['genre', 'country_of_origin']
        try:
            genre_categories = categorical_ohe_encoder.categories_[0].tolist()
            country_categories = categorical_ohe_encoder.categories_[1].tolist()
            
            # Define advisory categories manually, as they are not from the OHE encoder
            # This list MUST match the unique advisories found in your notebook's CAI cleaning
            # (e.g., from your `all_advisories_list` variable in the notebook)
            advisory_categories = ['Violence', 'Language', 'Sex', 'Horror', 'Alcohol', 'Crime', 'Drugs', 'Nudity', 'Occultism', 'Other', 'Parental Guidance', 'Profanity', 'Theme', 'Betting', 'Kissing', 'Fright', 'Mature', 'Harmful Imitable', 'Restricted', 'Suicide', 'Coarse Language', 'Obscenity', 'Horror/Scary', 'Weapon', 'Historical', 'Medical', 'Community', 'Gambling', 'LGBTQ', 'Content', 'UNSPECIFIED']
            advisory_categories = sorted(list(set(advisory_categories))) # Ensure unique and sorted
        except IndexError:
            st.error("Error: Categorical encoder categories might not match expected structure. Ensure 'genre' and 'country_of_origin' are being encoded in training script.")
            st.stop()

        genre = st.selectbox("Genre", options=genre_categories, index=genre_categories.index('drama') if 'drama' in genre_categories else 0)
        country_of_origin = st.selectbox("Country of Origin", options=country_categories, index=country_categories.index('United States') if 'United States' in country_categories else 0)

    with col2:
        classification_date = st.date_input("Classification Date", value=date.today())
        
        # Convert classification_date to year, month, day_of_week
        classification_year = classification_date.year
        classification_month = classification_date.month
        classification_day_of_week = classification_date.weekday() # Monday=0, Sunday=6

    st.subheader("Textual Information")
    synopsis = st.text_area("Synopsis", "A compelling story about a group of friends facing an unexpected challenge.")
    justification = st.text_area("Justification (e.g., reasons for classification)", "Contains mild thematic elements and some coarse language.")
    
    # Multi-select for Consumer Advisory Index
    selected_advisories = st.multiselect(
        "Consumer Advisory Index (Select all applicable)",
        options=advisory_categories, # Use the manually defined list
        default=['Violence', 'Language'] if 'Violence' in advisory_categories and 'Language' in advisory_categories else []
    )

    submitted = st.form_submit_button("Predict Rating")

    if submitted:
        st.subheader("Prediction Result")
        
        # --- Prepare input data for prediction ---
        
        # 1. Create a dictionary for the raw input features
        input_data = {
            'duration_mins': duration_mins,
            'classification_year': classification_year,
            'classification_month': classification_month,
            'classification_day_of_week': classification_day_of_week,
            'synopsis': synopsis,
            'justification': justification,
            'genre': genre,
            'country_of_origin': country_of_origin,
            'consumer_advisory_index': ','.join(selected_advisories) # Keep this for logging/display
        }
        
        # Create a DataFrame from the single input row
        input_df = pd.DataFrame([input_data])
        
        # 2. Apply text preprocessing (for TF-IDF)
        input_df['synopsis_processed'] = input_df['synopsis'].apply(preprocess_text)
        input_df['justification_processed'] = input_df['justification'].apply(preprocess_text)

        # 3. Generate TF-IDF features
        synopsis_tfidf_features = tfidf_synopsis.transform(input_df['synopsis_processed'])
        justification_tfidf_features = tfidf_justification.transform(input_df['justification_processed'])

        df_synopsis_tfidf = pd.DataFrame(synopsis_tfidf_features.toarray(),
                                         columns=['synopsis_tfidf_' + col for col in tfidf_synopsis.get_feature_names_out()],
                                         index=input_df.index)
        df_justification_tfidf = pd.DataFrame(justification_tfidf_features.toarray(),
                                              columns=['justification_tfidf_' + col for col in tfidf_justification.get_feature_names_out()],
                                              index=input_df.index)
        
        # 4. Generate One-Hot Encoded features for 'genre' and 'country_of_origin'
        categorical_input_df = input_df[['genre', 'country_of_origin']].copy()
        
        ohe_features = categorical_ohe_encoder.transform(categorical_input_df)
        df_ohe_features = pd.DataFrame(ohe_features.toarray(),
                                       columns=categorical_ohe_encoder.get_feature_names_out(['genre', 'country_of_origin']),
                                       index=input_df.index)

        # 5. Manually create dummy variables for 'consumer_advisory_index'
        cai_dummy_data = {}
        for adv in advisory_categories:
            col_name = f'CAI_{adv}'
            cai_dummy_data[col_name] = [1 if adv in selected_advisories else 0]
        df_cai_dummies = pd.DataFrame(cai_dummy_data, index=input_df.index)

        # --- SIMPLIFIED PREDICTION LOGIC: Pass raw features to the full pipeline (model) ---
        # Combine all features that would have been in X_train (raw, before the main preprocessor)
        # This includes numerical, TF-IDF, OHE for genre/country, and CAI dummies.
        
        # Start with numerical features
        X_for_pipeline = input_df[['duration_mins', 'classification_year', 'classification_month', 'classification_day_of_week']].copy()
        
        # Concatenate TF-IDF features
        X_for_pipeline = pd.concat([X_for_pipeline, df_synopsis_tfidf, df_justification_tfidf], axis=1)
        
        # Concatenate OHE features (for genre, country)
        X_for_pipeline = pd.concat([X_for_pipeline, df_ohe_features], axis=1)

        # Concatenate CAI dummy features
        X_for_pipeline = pd.concat([X_for_pipeline, df_cai_dummies], axis=1)

        # Ensure X_for_pipeline has all columns that X_train had, in the correct order.
        # Use X_columns_raw (which contains the names of the raw features expected by the pipeline).
        # This DataFrame will be the input to the 'model' (which is the full pipeline).
        final_input_df_for_pipeline = pd.DataFrame(0.0, index=X_for_pipeline.index, columns=X_columns_raw)
        
        for col in X_for_pipeline.columns:
            if col in final_input_df_for_pipeline.columns:
                final_input_df_for_pipeline.loc[:, col] = X_for_pipeline[col]
        
        # --- CRITICAL CHANGE: Pass DataFrame directly to model.predict() ---
        # The ColumnTransformer inside the pipeline expects a DataFrame input.
        prediction_encoded = model.predict(final_input_df_for_pipeline)[0] # <-- Pass DataFrame
        # --- END CRITICAL CHANGE ---

        predicted_rating = rating_order[prediction_encoded]

        st.success(f"The predicted rating for this film is: **{predicted_rating}**")

        st.markdown("---")
        st.write("Input Data Used for Prediction:")
        st.json(input_data)
        # Display the DataFrame that was passed to the model (before internal pipeline processing)
        st.write("Raw Features Passed to Pipeline (Sample of first 5 columns):")
        st.dataframe(final_input_df_for_pipeline.iloc[:, :5]) # Still display as DataFrame for user
