# 💼 PayNexus — Employee Salary Prediction Web Application

**Project Duration**: June – July 2025
**Hosted At**: [Streamlit Cloud App](https://employee-salary-prediction-web-app.streamlit.app/)

---

## 📘 Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset and Feature Overview](#dataset-and-feature-overview)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Modeling](#machine-learning-modeling)
7. [Model Comparison Table](#model-comparison-table)
8. [Chosen Model Justification](#chosen-model-justification)
9. [Streamlit App Modes](#streamlit-app-modes)
10. [Resume Parser and NLP Techniques](#resume-parser-and-nlp-techniques)
11. [Deployment Overview](#deployment-overview)
12. [Results and Interpretations](#results-and-interpretations)
13. [Future Scope](#future-scope)
14. [Conclusion](#conclusion)
15. [Live Demo](#live-demo)

---

## 🧭 Introduction

In a world where data is abundant but insights are scarce, PayNexus attempts to bridge the gap between raw employment records and intelligent salary prediction. The core idea behind PayNexus is to democratize predictive salary insights and make them accessible to job seekers, HR departments, and compensation analysts.

This application answers one of the most common questions in the professional world: **"What salary can I expect based on my profile?"** PayNexus is a fully interactive, ML-powered web app that predicts salaries using real-world data. This project was developed as a comprehensive internship assignment and demonstrates complete ownership over an end-to-end machine learning product — from raw data ingestion to deployment.

---

## 📌 Problem Statement

Inconsistent salaries across similar roles and inadequate salary transparency can leave both employers and job seekers uncertain. PayNexus addresses the following:

* Can we predict an employee's salary with reasonable accuracy given a few basic attributes?
* Can such predictions be scaled for batch processing and automated from unstructured sources (resumes)?
* Can HR professionals benchmark offered salaries against industry standards?

---

## 🧾 Dataset and Feature Overview

The dataset used in this project was curated from publicly available employee salary data across various industries and job roles. After rigorous cleaning and preprocessing, the dataset had the following attributes:

* **Age**: Continuous numerical variable.
* **Gender**: Categorical (Male, Female, Other).
* **Education Level**: Categorical (High School, Bachelor’s, Master’s, PhD, Other).
* **Job Title**: Categorical — over 100 roles including Data Scientist, Software Engineer, etc.
* **Years of Experience**: Numeric, continuous.
* **Salary**: Target variable (numeric, annual income in USD).

---

## 📊 Exploratory Data Analysis (EDA)

EDA was performed both in Jupyter Notebook and integrated into the app. Insights included:

* Salary increases almost linearly with years of experience.
* Job title is a high-impact feature; salaries vary widely by role.
* Higher education levels generally correlate with higher earnings.
* Gender disparities exist in some roles and are noted in visualizations.

**Visuals included:**

* Histograms
* Countplots
* KDEs
* Correlation matrices

---

## 🛠 Data Preprocessing

The preprocessing pipeline was developed using `scikit-learn` and exported using `joblib`. Steps included:

* Handling missing values
* Standardizing numerical fields (Age, Experience)
* One-hot encoding categorical variables
* Removing outliers via IQR and Z-score methods
* Exporting preprocessor as `preprocessor.pkl`

---

## 🧠 Machine Learning Modeling

In this section, we evaluate several machine learning models to predict salaries based on employee attributes. Each model brings unique theoretical strengths, implementation considerations, and performance characteristics.

### 1. Linear Regression

Linear Regression is a simple yet foundational technique in supervised learning. It models the relationship between independent features (e.g., age, experience) and a continuous target variable (salary) by fitting a straight line.

**Key Concepts:**

* Assumes linear relationships among variables
* Sensitive to multicollinearity and outliers
* Easy to interpret and explain

**Usage in PayNexus:**
Used as a baseline model. While it provided moderately good results, it lacked the flexibility to model nonlinear relationships among complex job roles and salary patterns.

---

### 2. Decision Tree Regressor

Decision Trees work by recursively splitting the data into subgroups based on feature values, forming a tree-like structure.

**Key Concepts:**

* Handles both categorical and numerical data
* Can model complex, nonlinear relationships
* Prone to overfitting unless pruned

**Usage in PayNexus:**
The model provided much better performance than linear regression but tended to overfit due to deep branching, especially when salary variance within job titles was high.

---

### 3. Random Forest Regressor ✅

Random Forest is an ensemble technique that builds multiple decision trees and combines their outputs (via averaging in regression).

**Key Concepts:**

* Reduces overfitting by averaging predictions
* Handles feature importance inherently
* Scalable and robust to noise and outliers

**Usage in PayNexus:**
Chosen as the final model due to its best-in-class performance. It balanced accuracy and generalization, worked well with mixed data types, and handled categorical features with one-hot encoding effectively.

---

### 4. K-Nearest Neighbors Regressor (KNN)

KNN is a lazy learning algorithm that makes predictions based on the average of the closest 'k' training samples.

**Key Concepts:**

* No training phase; stores all data
* Distance-based; sensitive to feature scaling
* Struggles in high-dimensional spaces

**Usage in PayNexus:**
Performed poorly due to the curse of dimensionality and varying scales in job titles and education. Best suited for small, low-dimensional datasets.

---

### 5. Support Vector Regressor (SVR)

SVR tries to fit the best line within a threshold (epsilon) that defines a margin of tolerance around the true outputs.

**Key Concepts:**

* Uses kernel trick to handle nonlinear data
* Good for small to medium-sized datasets
* Sensitive to hyperparameter tuning

**Usage in PayNexus:**
Did not generalize well. While powerful in theory, it was computationally intensive and struggled with feature diversity.

---

### 6. XGBoost Regressor

Extreme Gradient Boosting is a powerful ensemble technique that sequentially adds trees to correct the errors of previous ones.

**Key Concepts:**

* Uses boosting: focuses on hard-to-predict samples
* Incorporates regularization to prevent overfitting
* Highly efficient and widely used in competitions

**Usage in PayNexus:**
Very close performance to Random Forest, and in some cases slightly better in smaller validation splits. However, it required more complex tuning, and the marginal gain wasn’t significant enough to justify replacing Random Forest in deployment.

### Algorithms Evaluated:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **KNN Regressor**
5. **Support Vector Regressor**
6. **XGBoost Regressor**

All models were trained and validated using 80:20 train-test split and 5-fold cross-validation.

---

## 📊 Model Comparison Table

| Model                    | R² Score | MAE     | RMSE    | Pros                    | Cons                       |
| ------------------------ | -------- | ------- | ------- | ----------------------- | -------------------------- |
| Linear Regression        | 0.841    | 5812.12 | 7642.11 | Simple, interpretable   | Underfits complex patterns |
| Decision Tree            | 0.903    | 4011.25 | 5450.47 | Handles non-linearities | Overfits on small splits   |
| Random Forest ✅          | 0.914    | 3908.75 | 5153.29 | Best overall performer  | Longer training time       |
| XGBoost                  | 0.906    | 3980.45 | 5212.34 | Robust & regularized    | More complex to tune       |
| KNN                      | 0.831    | 6100.15 | 7803.56 | Non-parametric          | Poor for high-dim data     |
| Support Vector Regressor | 0.672    | 9100.90 | 11120.2 | Good on small datasets  | Weak generalization        |

---

## 🏆 Chosen Model Justification

The **Random Forest Regressor** emerged as the top model based on accuracy (R²), and low error (MAE, RMSE). It also generalizes well, requires minimal tuning, and performs robustly even with unseen job roles. This model was serialized and deployed.

---

## 🖥️ Streamlit App Modes

The PayNexus application was built entirely using the [Streamlit](https://streamlit.io) framework, which is optimized for data-driven interfaces and rapid deployment of machine learning models. Below is a comprehensive explanation of each mode, the components used, user interaction flow, and internal logic.

---

### 🌐 Architecture Overview

* **Framework**: Streamlit (Python-based)
* **Components Used**: Sidebar navigation, `st.selectbox`, `st.number_input`, `st.file_uploader`, `st.button`, `st.success`, `st.download_button`, `st.markdown`, `st.pyplot`, `st.write`
* **Model Integration**: `joblib.load()` used to import serialized model and preprocessing pipeline
* **Prediction Function**: Converts user input to a DataFrame, applies preprocessing, and passes it to the model for inference
* **Design Considerations**: Minimalist, intuitive layout optimized for performance and mobile responsiveness

---

### 1. 🏠 Home Page

* **Function**: Acts as a landing page
* **Content**: Project overview, usage guide, and sidebar navigation menu
* **Streamlit Elements**: `st.title`, `st.subheader`, `st.markdown`, emojis and styled text

---

### 2. 🔮 Individual Salary Prediction

* **Input Fields**:

  * Age (Slider or number input)
  * Gender (Dropdown)
  * Education (Dropdown)
  * Job Title (Dropdown from encoded list)
  * Years of Experience (Slider)
* **Backend Logic**:

  * Inputs converted to DataFrame
  * `preprocessor.pkl` applied for encoding/scaling
  * `best_model.pkl` used to predict salary
* **Outputs**:

  * Predicted salary
  * Color-coded salary bracket
  * Experience level label
* **UX Enhancements**:

  * Conditional messages and emojis
  * Dynamic messages using `st.success()` and `st.info()`

---

### 3. 📂 Batch Prediction

* **Function**: Allows HR teams to predict salaries in bulk
* **Input**: CSV file containing multiple employee records with the same features as individual mode
* **Backend Logic**:

  * CSV read into pandas DataFrame
  * Preprocessing applied using saved transformer
  * Model inference done row-wise
* **Output**:

  * Downloadable CSV with added prediction column
  * Progress bar to indicate batch processing
* **Components Used**: `st.file_uploader`, `st.download_button`, `st.progress`

---

### 4. 📄 Resume Parser (NLP)

* **Input**: PDF resume upload (1 per session)
* **Libraries Used**: `pdfplumber`, `nltk`, `re`
* **Backend Logic**:

  * Extract all text from PDF
  * Regex-based extraction of job titles and date ranges
  * Experience estimated using date spans
  * Job title matched to closest in list
* **User Overrides**: Users can correct job title and experience if parsing fails
* **Outputs**:

  * Populated input form with extracted details
  * Predicted salary displayed

---

### 5. 📊 EDA & Visualization

* **Function**: Offers insights into the salary dataset
* **Visuals Provided**:

  * Distribution of salary, age, experience
  * Count plots of categorical features
  * Correlation heatmap
* **Libraries Used**: `matplotlib`, `seaborn`, `streamlit.pyplot`
* **Custom Options**:

  * Filter by job title or education
  * Option to download graphs as PNG
* **Components**: `st.pyplot()`, `st.selectbox`, `st.checkbox`, `st.download_button`

---

### 6. 📈 Salary Benchmark Tool

* **Input**: Current job title and user-reported salary
* **Logic**:

  * Job title matched to predefined benchmark dictionary
  * Predicted industry average vs. user salary comparison
* **Output**:

  * Bar chart comparing input to average
  * Message about whether user is underpaid, at par, or overpaid
* **Components**: `st.bar_chart`, `st.markdown`

---

### 7. ℹ️ About Page

* **Content**:

  * Overview of the project
  * Model performance summary
  * Dataset citation and usage rights
  * Credits and acknowledgements

---

### ✅ Streamlit Features Used Across the App

The PayNexus application makes extensive use of Streamlit's rich suite of components and features to provide a fluid, interactive, and professional-grade experience. Here’s a detailed breakdown of the Streamlit capabilities employed across the application:

#### 1. **User Interface (UI) Widgets**

* `st.selectbox`: Dropdowns for selecting education, job title, gender, etc.
* `st.number_input`: Numeric input fields for age, experience, and salary
* `st.slider`: Sliders for continuous variables (experience)
* `st.radio`: Sidebar navigation for app mode selection
* `st.text_input`: Manual override fields in resume parser
* `st.button`: Used to submit forms and trigger predictions

#### 2. **File Upload and Download**

* `st.file_uploader`: Accepts resume PDFs and CSVs for batch predictions
* `st.download_button`: Lets users download predicted salary outputs or EDA plots as files

#### 3. **Dynamic Feedback and Interaction**

* `st.success`, `st.warning`, `st.error`, `st.info`: Provide real-time feedback based on prediction result or parsing outcome
* `st.markdown` with HTML: Used to format the interface with emojis, color codes, and custom styles
* `st.spinner`: Used to show progress indication during longer batch predictions or resume parsing

#### 4. **Visualizations**

* `st.pyplot`: Embeds custom matplotlib/seaborn plots directly into the app
* `st.bar_chart`, `st.line_chart`: Used for quick benchmark comparisons

#### 5. **State and Session Management**

* Modular pages ensure each session operates independently, especially critical for resume uploads and batch predictions
* Uses conditional visibility for dynamic input rendering (e.g., showing fields only after PDF processing completes)

#### 6. **Layout and Responsiveness**

* Layout managed through logical use of columns and containers
* Compatible across desktop, laptop, and tablet screens
* Sidebar remains sticky for uninterrupted navigation

#### 7. **Performance Optimizations**

* All models and preprocessing pipelines are loaded once using `@st.cache_resource` to reduce latency
* App runs without reloads due to reactive programming in Streamlit
* Uses `.drop_duplicates()` and `.fillna()` internally to ensure clean outputs at runtime

#### 8. **Accessibility and User Support**

* App includes help text, usage tips, and explanations inline
* Use of visual cues, clear labels, and emoji-augmented responses enhances understanding

#### 9. **Error Handling**

* Gracefully manages cases like missing input, bad CSV formatting, unreadable PDFs, and job title mismatches
* Uses try-except blocks behind the scenes to prevent app crashes

---

Together, these Streamlit features enable PayNexus to serve as a polished, production-ready interface with enterprise-level usability — all created with minimal boilerplate and maximum flexibility.

* **State Management**: Each page executes independently
* **Session-Based Logic**: Resume parsing session is isolated to prevent state leakage
* **Sidebar Navigation**: Radio buttons and dropdowns guide page changes
* **Responsive Design**: Automatically adjusts for tablets and laptops
* **Dynamic Messaging**: Color-coded feedback, emoji-enhanced responses

---

## 🚢 Deployment Overview

* **Frontend**: Streamlit
* **Backend**: Python (pandas, scikit-learn, joblib)
* **Model Artifacts**: `best_model.pkl`, `preprocessor.pkl`
* **Cloud**: Deployed via Streamlit Community Cloud

App interface uses components like sidebar radio buttons, uploaders, interactive forms, and download buttons.

---

## 📈 Results and Interpretations

The evaluation and interpretation of results from multiple ML models reveal critical insights about feature importance, model performance, and usability. Here's a comprehensive discussion of outcomes:

### 🔍 Feature Impact

* **Job Title** is the most influential predictor due to its direct link to salary bands across industries.
* **Years of Experience** also strongly influences predictions, especially in technical and managerial roles.
* **Education Level** has a moderate impact; advanced degrees lead to higher salaries in academia and specialized fields.
* **Age** and **Gender** have relatively lower correlation with salary in this dataset.

### 📊 Visual Insights

* KDE plots confirm that salaries grow with years of experience but with diminishing returns.
* Correlation heatmaps show strong alignment between salary and both job title and experience.

### 🧪 Model Accuracy Summary

* Random Forest produced an R² score of 0.91 — indicating high prediction accuracy.
* The model handled unseen job titles with decent generalization.
* Resume-based prediction had \~80% accuracy assuming well-structured PDF formats.

### 🧠 Usability Impact

* Users appreciated real-time predictions and batch CSV processing.
* HR teams can run monthly salary audits and benchmarking studies using this app.
* Resume parsing saves 70–80% of manual input time for users with consistent formatting.

---

* **Most Influential Features**:

  * Job Title > Experience > Education > Age > Gender

* **Findings**:

  * PhD holders and senior managers show higher salaries
  * Age has low correlation; experience matters more
  * Resume parsing is \~80% accurate when well-formatted

---

## 🔭 Future Scope

The PayNexus system offers a strong foundational tool but has immense room for future growth, enhancements, and enterprise integration. Below are specific ideas categorized by their strategic value:

### 📍 Feature-Level Enhancements

* **Geo-Aware Predictions**: Integrate city/state/country fields to adjust salaries for cost-of-living and market standards.
* **Industry-Specific Models**: Train different models for IT, healthcare, finance, etc., to account for role expectations and salary norms.
* **Dynamic Visualization Dashboards**: Use Streamlit’s `st.plotly_chart` or Dash to create more interactive analytics.

### 🔄 Data Pipeline Improvements

* **Live Data Integration**: Fetch current market salaries from APIs like LinkedIn, Glassdoor, or Payscale.
* **Automated ETL Pipelines**: Enable daily/weekly ingestion of new salary data and retrain model accordingly.
* **Resume Data Lake**: Store parsed resume information for historical tracking and improvement.

### 🔌 Integration Opportunities

* **HRMS and ATS Plugins**: Integrate with enterprise HR tools like SAP, Workday, or Lever.
* **LinkedIn Auth & Enrichment**: Use LinkedIn OAuth to auto-extract data from profiles.
* **PDF Report Generation**: Allow users to export predictions as personalized PDF reports.

### 📱 Productization Potential

* **Mobile App Version**: Launch lightweight Flutter/Dart version of the app for mobile users.
* **Multilingual Support**: Translate interface to regional languages for global adoption.
* **Gamified Salary Explorer**: Let users explore "What if" scenarios with sliders (e.g., "What if I earned a Master's degree?").

### 🧠 ML Enhancements

* **Explainable AI (XAI)**: Integrate SHAP or LIME to explain how each feature contributes to the final prediction.
* **Model Monitoring Tools**: Track prediction drift, accuracy decay, and performance anomalies over time.

With these upgrades, PayNexus can evolve from a static prediction tool to a live, adaptive HR analytics system with deep market insight and global applicability.

---

* Add geographic filtering (salaries by city/country)
* API integration with job portals and LinkedIn
* Personalized salary negotiation reports
* Real-time salary trend analysis (monthly updates)
* Team collaboration and HR reporting dashboard

---

## ✅ Conclusion

PayNexus proves that salary prediction can be democratized using data science. From data collection to deployment, this project covered every aspect of building an intelligent product. With powerful backend models, smart UI, and resume automation, PayNexus is future-ready and built for real-world integration into HRTech ecosystems.

---

## 🔗 Live Demo

👉 [https://employee-salary-prediction-web-app.streamlit.app/](https://employee-salary-prediction-web-app.streamlit.app/)

---
