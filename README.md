# 🚀 Automating Classification of Audio-Visual Content & Rating  

![Banner](images/film_classification_board.jpg)

<br>

<p align="left">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://audio-visual-content-classification-ctzt6xe4trmmcfdfqk944g.streamlit.app)
[![GitHub stars](https://img.shields.io/github/stars/Edwinkorir38/audio-visual-content-classification?style=social)](https://github.com/Edwinkorir38/audio-visual-content-classification)
[![License](https://img.shields.io/github/license/Edwinkorir38/audio-visual-content-classification)](LICENSE)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)

</p>


## 📌 Author 
- Edwin Korir

---

## 🧭 Table of Contents  
1. [Project Overview](#1-project-overview)  
2. [Business Understanding](#2-business-understanding)  
3. [Domain of Application](#3-domain-of-application)  
4. [Dataset Overview](#4-dataset-overview)  
5. [Data Preparation](#5-data-preparation)  
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)  
7. [Modeling](#7-modeling)  
8. [Evaluation & Results](#8-evaluation--results)  
9. [Conclusion](#9-conclusion)  
10. [Recommendations](#10-recommendations)  
11. [Next Steps](#11-next-steps)  
12. [Use Cases](#12-use-cases)  
13. [Future Work](#13-future-work)  
14. [Ethical Considerations](#14-ethical-considerations)  
15. [Deployment](#15-deployment)  
16. [Acknowledgements](#16-acknowledgements)  

---

## 1. Project Overview  
This project builds a machine-learning and natural-language-processing pipeline to automatically classify audiovisual content (films, series, online videos) based on age-classification ratings (GE, PG, 16, 18, R). It assists regulatory bodies (e.g., Kenya Film Classification Board), parents, educators, and streaming platforms by enhancing content-review speed, objectivity, and scalability.

---

## 2. Business Understanding  
### Context  
With the explosion of digital content across platforms such as YouTube, TikTok and streaming services, the manual classification of each piece of content for age-appropriateness is no longer scalable.  
The Kenya Film Classification Board uses guidelines categorising content into GE, PG, 16, 18 and Restricted.

### Business Problem  
- Manual classification is time-consuming and subjective.  
- Regulatory bodies face increasing content volume.  
- Lack of fast feedback tools for parents and platforms.

### Project Goals  
- Build a supervised classification model that predicts ratings.  
- Accelerate regulatory film-review workflow.  
- Enable platforms to implement self-regulation and compliance.  
- Provide parental control and safer content filters.  
- Promote visibility of local film classification insights.

---

## 3. Domain of Application  
This solution sits at the intersection of:  
- **Regulatory Tech** (governance automation)  
- **Media & Entertainment** (content tagging & filtering)  
- **Machine Learning** (multiclass classification)  
- **Natural Language Processing** (synopsis & justification analysis)  
- **Education & Parental Tools** (safer digital experiences)

### Stakeholders & Benefits  
| Stakeholder           | Value Delivered                              |
|------------------------|---------------------------------------------|
| Regulators (KFCB)      | First-layer automated classification support |
| Parents & Educators    | Smarter, age-appropriate content filters    |
| Streaming Platforms    | Automated tagging & compliance workflows    |
| Content Creators       | Guidance on target audience & compliance    |

---

## 4. Dataset Overview  
This dataset contains film records classified during FY 2022/2023 to FY 2024/2025 by KFCB.  
### Sample Fields  
| Column Name         | Description |
|----------------------|-------------|
| `film_title`         | Name of the content (e.g., *The Lion King*) |
| `class_of_film`      | Format: Feature, Series, Short, Documentary |
| `genre`              | Genre category: Drama, Action, Animation, etc. |
| `synopsis`           | Summary of the storyline (key text feature) |
| `justification`      | Written reasons for classification |
| `rating`             | Target label: GE, PG, 16, 18, R |
| `cai`                | Consumer Advisory Index (flags for violence, language, etc.) |
| `duration_mins`      | Length of content in minutes |
| `date_classified`    | Date of official classification |
| `country_of_origin`  | Production country |
| `platform`           | Distribution medium (YouTube, TV, Cinema, etc.) |

---

## 5. Data Preparation  
Key pre-processing steps included:  
- Standardising column names and types  
- Filling missing numeric and categorical values  
- Parsing `date_classified` into `classification_year`, `month`, `day_of_week`  
- Creating features: `title_length`, `description_word_count`  
- Extracting text features via TF-IDF on `synopsis` and `justification`  
- One-hot encoding `genre` and `country_of_origin`, creating dummy variables for CAI flags  

---

## 6. Exploratory Data Analysis (EDA)  
### Rating Distribution
![Rating Distribution](images/Distribution_of_film_ratings.png)  

- Majority of content is classified as **GE** and **PG**
- Very few items fall under the **Restricted** category  
### Top Genres vs Ratings  
![Genres vs Rating](images/Film_ratings_distribution_across_top_10_genres.png)

* This grouped bar chart shows how film ratings (GE, PG, 16, 18, R) are distributed across the top 10 genres. It highlights which genres are more family-friendly and which tend to have restrictive ratings. For example, genres like horror or action may have more 18+ ratings, while animation and documentary often align with general audience categories. The chart offers a clear view of how age-appropriateness varies by genre.

**Insights**
* Drama dominates with a wide spread of ratings, especially PG and 16.

* Documentary and Animation are mostly family-friendly (GE, PG).

* Action leans toward more mature ratings (16, 18).

* Commercials are mostly rated GE.

* Other genres like Reality and Comedy have mixed but generally lighter ratings.
### Ratings vs Country of Origin  
![Ratings vs Country of Origin](images/Film_ratings_distribution_across_top_10_countries_of_origin.png)

This grouped bar chart compares film rating distributions across the top 10 countries of origin. Each country shows a breakdown of ratings (GE, PG, 16, 18, R), revealing patterns such as:

* Countries that produce more family friendly vs. restrictive content

* Cultural or regulatory differences in rating practices

* Regional tendencies in film classification (e.g., more R-rated films from some countries)

* The visualization helps identify how content standards and audience targeting vary internationally.

**Insights**

* The chart shows that Kenya leads in film production across all rating categories, especially in GE and PG, indicating family-friendly content. The United States has a more balanced distribution with more 16–R rated films, suggesting mature content. India leans heavily toward 16-rated films. Other countries like Brazil, Tanzania, and Uganda mostly produce general audience films. Overall, the chart highlights regional differences in content maturity and classification.
 
### Text Analysis `synopsis` and `justification`
- Objective: Gain insights from free-text fields by visualizing common words and identifying key themes. 
- For basic EDA, we focused on word frequency and visualization using word clouds. For more advanced modeling, we typically used TfidfVectorizer or CountVectorizer to convert text into numerical features.

![Word cloud for Film Synopsis](images/word_cloud_for_film_synopsis.png)

![Word cloud for Film Justifications](images/word_cloud_for_classification_justifications.png)

![Top 20 Words in Film Synopsis](images/top_20_most_common_words_in_film_synopses.png)

![Top 20 Words in Film Justifications](images/top_20_most_common_words_in_classification_justification.png)

**Justifications**

These bar charts highlight the key language patterns in film descriptions and rating explanations:

* Film Synopses emphasize recurring themes, actions, and characters—revealing common narrative trends.

* Classification Justifications spotlight frequent concerns like violence, language, or sexual content—indicating what regulators focus on when assigning ratings.

Together, they contrast creative storytelling with regulatory reasoning.

### Exploring length of the Synopsis

![Length of the Synopsis](images/distribution_of_synopsis_length.png)

**Justifications**

The synopsis length distribution shows how detailed film descriptions typically are:

* Most synopses fall within a moderate length range.

* There's noticeable skewness, with some very short or very long entries.

* Short synopses may lack detail, while long ones could indicate data inconsistencies.

This insight helps tailor text analysis methods by revealing the need to handle varying text lengths appropriately.

---

### Type of Problem

- **Multiclass classification** using both structured (duration, genre, platform) and unstructured (synopsis) data

### ML Algorithms Used


| Model                                 | Strength                                                                 |
|---------------------------------------|--------------------------------------------------------------------------|
| **Logistic Regression**               | Serves as a simple, interpretable baseline for classification tasks      |
| **Decision Tree Classifier**          | Easy to interpret, handles both numerical and categorical data           |
| **Random Forest Classifier**          | Reduces overfitting, handles feature interactions well                   |
| **Gradient Boosting - XGBoost**       | High performance, effective for imbalanced datasets and ranking features |
| **Gradient Boosting - LightGBM**      | Fast training speed and efficient memory usage on large datasets         |
| **Naive Bayes - MultinomialNB**       | Strong with high-dimensional, text-heavy data (e.g., TF or TF-IDF inputs)|


---

##  8. Evaluation & Results

### Metrics used

- Accuracy
- Precision / Recall / F1 Score
- Confusion Matrix

### Model Evaluation Summary

Below are the results of GridSearchCV and evaluation metrics for each classification algorithm used in the project:



#### **1. Logistic Regression**
- **Best Parameters**: `C=1`, `solver='lbfgs'`
- **Accuracy**: 0.69  
- **F1-Weighted Score (CV)**: 0.69  
- **Notes**: Acts as a strong baseline model. Performs well for common classes like 'GE' and 'PG', but struggles to generalize to rare classes like 'R'.



#### **2. Decision Tree Classifier**
- **Best Parameters**: `max_depth=None`, `min_samples_leaf=1`
- **Accuracy**: 0.70  
- **F1-Weighted Score (CV)**: 0.71  
- **Notes**: Captures class boundaries well, especially for 'GE', 'PG', and '18'. However, the model shows signs of overfitting on training data.



#### **3. Random Forest Classifier**
- **Best Parameters**: `n_estimators=200`, `max_depth=20`
- **Accuracy**: 0.76  
- **F1-Weighted Score (CV)**: 0.76  
- **Notes**: Top performer overall. Achieves excellent balance across all major classes, making it suitable for production-level tasks.



#### **4. XGBoost**
- **Best Parameters**: `learning_rate=0.1`, `n_estimators=200`
- **Accuracy**: 0.77  
- **F1-Weighted Score (CV)**: 0.76  
- **Notes**: Strong recall for the 'PG' class. Performs efficiently but slightly underpredicts the '18' class. Great trade-off between performance and speed.



#### **5. LightGBM**
- **Best Parameters**: `learning_rate=0.1`, `n_estimators=200`
- **Accuracy**: 0.75  
- **F1-Weighted Score (CV)**: 0.75  
- **Notes**: Fast and scalable alternative to XGBoost. Provides competitive results and is efficient for large-scale or streaming applications.



#### **6. Multinomial Naive Bayes**
- **Best Parameters**: `alpha=1.0`
- **Accuracy**: 0.75  
- **F1-Weighted Score (CV)**: 0.75  
- **Notes**: Works particularly well with text data, especially when using TF-IDF features. Performs solidly on the '16' class but underrepresents rare classes.


### General Observations

- **Top Models**: Random Forest, XGBoost, and LightGBM consistently show the best accuracy and F1 scores.
- **Challenging Classes**: The rare class 'R' remains difficult to predict across all models due to class imbalance.
- **Recommendation**: Ensemble methods (Random Forest, XGBoost) are preferred for deployment due to robustness and generalizability.




###  Model Evaluation Summary

| **Model**                  | **Best Parameters**                                 | **Accuracy** | **F1-Weighted (CV)** | **Notes**                                                                 |
|---------------------------|-----------------------------------------------------|--------------|----------------------|---------------------------------------------------------------------------|
| **Logistic Regression**    | `C=1`, `solver='lbfgs'`                              | 0.69       | 0.69              | Solid baseline; good for 'GE'/'PG'; weak on rare classes like 'R'         |
| **Decision Tree**          | `max_depth=None`, `min_samples_leaf=1`              | 0.70       | 0.71               | Good for 'GE'/'PG', handles '18'; tends to overfit                        |
| **Random Forest**          | `n_estimators=200`, `max_depth=20`                  | 0.76       | 0.76                balanced across most classes                      |
| **XGBoost**                | `learning_rate=0.1`, `n_estimators=200`             | 0.77      | 0.76              | Strong on 'PG'; slightly lower on '18'; efficient                         |
| **LightGBM**               | `learning_rate=0.1`, `n_estimators=200`             | 0.75      | 0.75               | Fast, efficient; similar performance to XGBoost                           |
| **Multinomial Naive Bayes**| `alpha=1.0`                                          | 0.75       | 0.75              | Great with text data; solid for '16' class                                |



### Confusion Matrix for best performing model

![Confusion Matrix](images/xg_boost.png)



## XGBoost – Best Model Insights (Confusion Matrix)
* Excellent on common classes: Achieved high precision and recall for 'GE' (148/152) and 'PG' (138/169).

* Handles mid-range classes fairly well: Reasonable performance on '16' and '18', though some overlap exists.

* Struggles with rare class 'R': Failed to correctly predict the only 'R' instance—common across all models.

* Best overall balance: Delivered the highest accuracy (0.77) with strong F1 score and robust class separation, making it the most reliable model for deployment.

## Best Performing Model
### XGBoost Classifier

* Accuracy: 76.50%

* F1-Weighted Score (Cross-Validation): 0.7457

* Strong overall performance across most classes, especially 'GE' and 'PG'

* Efficient and scalable; handles feature interactions well

* Slight confusion between '16' and '18', but superior generalization overall


---

## 9. Conclusion  
 * Successfully built a machine learning model to classify films based on age-appropriateness using KFCB guidelines.
  
  * Random Forest performed best (Accuracy: 77.48%, F1: 0.7472).
  
  * Text features like synopses and justifications were key in improving prediction.
  
  * EDA revealed rating patterns across genres, platforms, and countries.
  
  * The solution supports regulators, parents, and content platforms in faster, scalable, and objective classification.

  * Deployed the model using streamlit

 ## Project Challenges
* Missing Data: Key columns like VENUE and CONTACT had many null values.

* Data Cleaning: Inconsistent formats in fields like DURATION(MINS) required extensive preprocessing.

* Class Imbalance: Rare ratings like 'R' had very few samples, hurting model recall.

* Similar Class Overlap: Models confused PG, 16, and 18 due to feature similarity.

* Text Feature Complexity: High-dimensional TF-IDF features from SYNOPSIS increased model complexity.

* Evaluation Limitation: Low support for rare classes affected confusion matrix reliability.
 

---

## 10. Recommendations  
- Use model as a pre-screening tool for regulators.  
- Integrate with platforms for automatic tagging.  
- Explore transformer-based models for text.  
- Build APIs for real-time classification.  
- Include multimodal features (image, audio).  
- Conduct fairness audits on classification outputs.  

---

## 11. Next Steps  
- Deploy transformers (BERT/RoBERTa) for synopsis processing.  
- Integrate image/audio analysis for expanded coverage.  
- Develop dashboard for regulators + parents.  
- Implement human-in-loop feedback.  
- Expand to other languages and markets.  

---

## 12. Use Cases  
| Use Case                     | Impact                                           |
|-----------------------------|--------------------------------------------------|
| Content Classification      | Speeds regulator review, reduces backlog        |
| Parental Controls           | Makes content safer for children                |
| Platform Integration        | Enables real-time compliance tagging            |
| Local Film Promotion        | Highlights Kenyan-origin content                |

---

## 13. Future Work  
- Implement transformer-NLP modelling  
- Develop a classification API + live dashboard  
- Extend to image/audio multimodal analysis  
- Address bias and under-represented classes  
- Introduce personalization & filtering tools  

---

## 14. Ethical Considerations  
- Model may inherit classification bias from historical labels  
- Transparency required for end-users on how ratings are assigned  
- Privacy: Scraped or annotated data must be handled securely  
- Regular audits and monitoring of classification fairness  

---

## 15. Deployment  
### 🔗 Live App  
[View the live app →](https://audio-visual-content-classification-ctzt6xe4trmmcfdfqk944g.streamlit.app/)

### 🧰 Run Locally  
```bash
git clone https://github.com/<YOUR-USERNAME>/audio-visual-content-classification.git
cd audio-visual-content-classification
python3 -m venv venv
source venv/bin/activate            # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 16. Acknowledgements

- Kenya Film Classification Board (KFCB)
- OpenAI, Scikit-learn, Matplotlib, Pandas, XGBoost
- My teammates and mentors in the Data Science community

## 👤 Author  

<details open>
<summary><h3 style="display: inline;">✨ About the Author (Click to Expand)</h3></summary>
<br>

<div align="center">

### **Edwin Korir**  
Machine Learning • NLP • Data Science  

[![Email](https://img.shields.io/badge/Email-Contact%20Me-red?logo=gmail&logoColor=white)](mailto:ekorir99@gmail.com)  
[![GitHub](https://img.shields.io/badge/GitHub-Edwinkorir38-black?logo=github)](https://github.com/Edwinkorir38)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/edwin-korir-90a794382)  
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--9153--0080-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0007-9153-0080)  
[![Portfolio](https://img.shields.io/badge/Website-Portfolio-blueviolet?logo=vercel&logoColor=white)](https://github.com/Edwinkorir38)  
[![Twitter](https://img.shields.io/badge/Twitter-Follow-black?logo=x&logoColor=white)](https://twitter.com/ekorir99)

</div>

</details>
