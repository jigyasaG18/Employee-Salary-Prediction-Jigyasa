import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pdfplumber
import re
import nltk
from datetime import datetime

job_salary_benchmarks = {
    'Software Engineer': 85000,
    'Data Analyst': 65000,
    'Senior Manager': 110000,
    'Sales Associate': 45000,
    'Director': 130000,
    'Marketing Analyst': 70000,
    'Product Manager': 105000,
    'Sales Manager': 90000,
    'Marketing Coordinator': 55000,
    'Senior Scientist': 120000,
    'Software Developer': 80000,
    'HR Manager': 75000,
    'Financial Analyst': 70000,
    'Project Manager': 95000,
    'Customer Service Rep': 40000,
    'Operations Manager': 90000,
    'Marketing Manager': 85000,
    'Senior Engineer': 105000,
    'Data Entry Clerk': 35000,
    'Sales Director': 120000,
    'Business Analyst': 80000,
    'VP of Operations': 180000,
    'IT Support': 60000,
    'Recruiter': 65000,
    'Financial Manager': 95000,
    'Social Media Specialist': 55000,
    'Software Manager': 110000,
    'Junior Developer': 60000,
    'Senior Consultant': 115000,
    'Product Designer': 90000,
    'CEO': 250000,
    'Accountant': 70000,
    'Data Scientist': 115000,
    'Marketing Specialist': 60000,
    'Technical Writer': 65000,
    'HR Generalist': 60000,
    'Project Engineer': 85000,
    'Customer Success Rep': 50000,
    'Sales Executive': 70000,
    'UX Designer': 85000,
    'Operations Director': 130000,
    'Network Engineer': 80000,
    'Administrative Assistant': 40000,
    'Strategy Consultant': 110000,
    'Copywriter': 55000,
    'Account Manager': 70000,
    'Director of Marketing': 140000,
    'Help Desk Analyst': 50000,
    'Customer Service Manager': 65000,
    'Business Intelligence Analyst': 90000,
    'Event Coordinator': 45000,
    'VP of Finance': 190000,
    'Graphic Designer': 55000,
    'UX Researcher': 85000,
    'Social Media Manager': 65000,
    'Director of Operations': 150000,
    'Senior Data Scientist': 130000,
    'Junior Accountant': 50000,
    'Digital Marketing Manager': 80000,
    'IT Manager': 100000,
    'Customer Service Representative': 40000,
    'Business Development Manager': 90000,
    'Senior Financial Analyst': 95000,
    'Web Developer': 75000,
    'Research Director': 140000,
    'Technical Support Specialist': 60000,
    'Creative Director': 130000,
    'Senior Software Engineer': 115000,
    'Human Resources Director': 130000,
    'Content Marketing Manager': 75000,
    'Technical Recruiter': 70000,
    'Sales Representative': 55000,
    'Chief Technology Officer': 230000,
    'Junior Designer': 45000,
    'Financial Advisor': 70000,
    'Junior Account Manager': 48000,
    'Senior Project Manager': 105000,
    'Principal Scientist': 140000,
    'Supply Chain Manager': 95000,
    'Senior Marketing Manager': 110000,
    'Training Specialist': 60000,
    'Research Scientist': 90000,
    'Junior Software Developer': 60000,
    'Public Relations Manager': 80000,
    'Operations Analyst': 65000,
    'Product Marketing Manager': 90000,
    'Senior HR Manager': 90000,
    'Junior Web Developer': 60000,
    'Senior Project Coordinator': 90000,
    'Chief Data Officer': 220000,
    'Digital Content Producer': 60000,
    'IT Support Specialist': 60000,
    'Senior Marketing Analyst': 95000,
    'Customer Success Manager': 70000,
    'Senior Graphic Designer': 75000,
    'Software Project Manager': 110000,
    'Supply Chain Analyst': 70000,
    'Senior Business Analyst': 95000,
    'Junior Marketing Analyst': 48000,
    'Office Manager': 50000,
    'Principal Engineer': 130000,
    'Junior HR Generalist': 48000,
    'Senior Product Manager': 115000,
    'Junior Operations Analyst': 48000,
    'Senior HR Generalist': 75000,
    'Sales Operations Manager': 90000,
    'Senior Software Developer': 110000,
    'Junior Web Designer': 48000,
    'Senior Training Specialist': 75000,
    'Senior Research Scientist': 130000,
    'Junior Sales Representative': 45000,
    'Junior Marketing Manager': 55000,
    'Junior Data Analyst': 48000,
    'Senior Product Marketing Manager': 115000,
    'Junior Business Analyst': 48000,
    'Senior Sales Manager': 100000,
    'Junior Marketing Specialist': 48000,
    'Junior Project Manager': 50000,
    'Senior Accountant': 85000,
    'Director of Sales': 130000,
    'Junior Recruiter': 48000,
    'Senior Business Development Manager': 110000,
    'Senior Product Designer': 100000,
    'Junior Customer Support Specialist': 45000,
    'Senior IT Support Specialist': 80000,
    'Junior Financial Analyst': 48000,
    'Senior Operations Manager': 105000,
    'Director of Human Resources': 140000,
    'Junior Software Engineer': 60000,
    'Senior Sales Representative': 75000,
    'Director of Product Management': 140000,
    'Junior Copywriter': 45000,
    'Senior Marketing Coordinator': 85000,
    'Senior Human Resources Manager': 110000,
    'Junior Business Development Associate': 48000,
    'Senior Account Manager': 95000,
    'Senior Researcher': 110000,
    'Junior HR Coordinator': 48000,
    'Director of Finance': 150000,
    'Junior Marketing Coordinator': 48000,
    'Junior Data Scientist': 60000,
    'Senior Operations Analyst': 95000,
    'Senior Human Resources Coordinator': 90000,
    'Senior UX Designer': 90000,
    'Junior Product Manager': 55000,
    'Senior Marketing Specialist': 90000,
    'Senior IT Project Manager': 110000,
    'Senior Quality Assurance Analyst': 90000,
    'Director of Sales and Marketing': 150000,
    'Senior Account Executive': 95000,
    'Director of Business Development': 150000,
    'Junior Social Media Manager': 48000,
    'Senior Human Resources Specialist': 90000,
    'Senior Data Analyst': 95000,
    'Director of Human Capital': 150000,
    'Junior Advertising Coordinator': 45000,
    'Junior UX Designer': 48000,
    'Senior Marketing Director': 140000,
    'Senior IT Consultant': 110000,
    'Senior Financial Advisor': 95000,
    'Junior Business Operations Analyst': 48000,
    'Junior Social Media Specialist': 48000,
    'Senior Product Development Manager': 115000,
    'Junior Operations Manager': 50000,
    'Senior Software Architect': 130000,
    'Junior Research Scientist': 48000,
    'Senior Financial Manager': 110000,
    'Senior HR Specialist': 90000,
    'Senior Data Engineer': 115000,
    'Junior Operations Coordinator': 45000,
    'Director of HR': 140000,
    'Senior Operations Coordinator': 95000,
    'Junior Financial Advisor': 45000,
    'Director of Engineering': 150000,
    'Software Engineer Manager': 120000,
    'Back end Developer': 80000,
    'Senior Project Engineer': 110000,
    'Full Stack Engineer': 95000,
    'Front end Developer': 80000,
    'Developer': 80000,
    'Front End Developer': 80000,
    'Director of Data Science': 160000,
    'Human Resources Coordinator': 55000,
    'Junior Sales Associate': 45000,
    'Human Resources Manager': 75000,
    'Juniour HR Generalist': 48000,
    'Juniour HR Coordinator': 48000,
    'Digital Marketing Specialist': 60000,
    'Receptionist': 35000,
    'Marketing Director': 140000,
    'Social M': 55000,
    'Social Media Man': 55000,
    'Delivery Driver': 35000,
}

# Download punkt tokenizer (only once)
nltk.download('punkt')

def parse_experience(text):
    # Regex to find date ranges e.g. June 2025 – Present or May 2024 – July 2024 or 2022 – 2026
    date_ranges = re.findall(
        r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})\s*[-–]\s*(\b(?:Present|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\bPresent|\d{4})', 
        text, re.IGNORECASE)

    total_months = 0
    now = datetime.now()

    for start_str, end_str in date_ranges:
        try:
            start = datetime.strptime(start_str, '%B %Y')
        except ValueError:
            try:
                start = datetime.strptime(start_str, '%b %Y')
            except:
                continue
        
        if end_str.lower() == 'present':
            end = now
        else:
            try:
                end = datetime.strptime(end_str, '%B %Y')
            except ValueError:
                try:
                    end = datetime.strptime(end_str, '%b %Y')
                except:
                    try:
                        end = datetime.strptime(end_str, '%Y')
                        end = end.replace(month=12, day=31)
                    except:
                        end = now
        
        diff = (end.year - start.year) * 12 + (end.month - start.month)
        if diff > 0:
            total_months += diff

    total_years_exp = round(total_months / 12, 1)
    return total_years_exp

def extract_job_title(text):
    # Try to extract WORK EXPERIENCE or EXPERIENCE section
    work_exp_section = re.search(
        r'(?:WORK EXPERIENCE|EXPERIENCE)(.*?)(?:PROJECTS|CERTIFICATIONS|ACHIEVEMENTS|$)', 
        text, re.DOTALL | re.IGNORECASE
    )
    job_title = "Unknown"
    
    if work_exp_section:
        work_text = work_exp_section.group(1)
        if work_text:
            for line in work_text.splitlines():
                line = line.strip()
                if re.search(r'(engineer|analyst|manager|director|associate|intern|fellow|consultant|developer|software)', line, re.IGNORECASE):
                    job_title = line
                    break
            else:
                # fallback: first non-empty line in work_text
                for line in work_text.splitlines():
                    if line.strip():
                        job_title = line.strip()
                        break
    else:
        # No WORK EXPERIENCE or EXPERIENCE section found
        job_title = "Unknown (Could not detect job title)"

    return job_title

# Load models and default dataset
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
default_df = pd.read_csv('salary_data.csv')

gender_options = default_df['Gender'].dropna().unique().tolist()
education_options = default_df['Education Level'].dropna().unique().tolist()
job_title_options = default_df['Job Title'].dropna().unique().tolist()

st.sidebar.title("OPTIONS")
st.sidebar.markdown("**Choose Mode:**")
option = st.sidebar.radio(label="", options=( 
                          ("Home", "Individual Prediction", "Batch Prediction", "EDA & Visualization", "Resume Parser" , "Salary Comparison Tool", "About")))

# Reduce space by adding negative margin via markdown hack 
st.sidebar.markdown("<style>div[data-testid='stRadio'] > label {margin-top: -30px;}</style>", unsafe_allow_html=True)
if option == "Home":
    st.markdown("""
    <h1 style='text-align: center; text-transform: uppercase; color: #2F4F4F; margin-bottom: 0;'>
        💼 PayNexus
    </h1>
    <h3 style='text-align: center; color: #4B6E8E; margin-top: 0; font-weight: bold;'>
        Your Own Salary Predictor App
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("""
<h2 style='text-align: center; font-weight: bold; line-height: 1.2; margin-top: 0;'>
    🏡Welcome to Employee Salary<br>
    Prediction App
</h2>
""", unsafe_allow_html=True)


    st.markdown("""
    ### 👋 Get Started
    Please select one of the options from the **sidebar** to:

    - 🔮 **Predict Individual Salaries:** Enter personal details to get a tailored salary estimate.
    - 📂 **Batch Predictions:** Upload CSV files for bulk salary predictions efficiently.
    - 📊 **EDA & Visualization:** Explore salary trends, distributions, and insights with interactive charts.
    - 📄 **Resume Parser:** Upload your resume (PDF) to extract info and predict your salary.
    - 📈 **Salary Comparison Tool:** Compare your current salary with industry standards and peers.
    - ⚙️ **Model Info:** Learn about the machine learning model powering the predictions.
    - 🛠️ **Settings:** Customize preferences for your prediction experience.
    - ℹ️ **About the App:** Get detailed info on how the app works and its data sources.

    ---
    🧠 This app is powered by **Machine Learning**, trained on over 6700 real employee records, and built with **Streamlit** for an interactive experience.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    
elif option == "Individual Prediction":
    st.markdown("""
    <h1 style='text-align: center; text-transform: uppercase;'>
      🔮 Employee Salary Predictor
    </h1>
    <h2 style='text-align: center; text-transform: uppercase; font-weight: bold;'>
        Individual Prediction
    </h2>
    """, unsafe_allow_html=True)
    st.markdown("##### 📝 Fill in the details below to predict your salary:")

    with st.form("input_form"):
        age = st.number_input("🧍 Age", 18, 70, 30)
        gender = st.selectbox("🚻 Gender", gender_options)
        education = st.selectbox("🎓 Education Level", education_options)
        job_title = st.selectbox("💼 Job Title", sorted(job_title_options))
        years_exp = st.number_input("📈 Years of Experience", 0, 50, 5)

        submitted = st.form_submit_button("✨ Predict Salary")

    if submitted:
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [job_title],
            "Years of Experience": [years_exp]
        })

        try:
            input_transformed = preprocessor.transform(input_df)
            prediction = model.predict(input_transformed)[0]

            st.markdown("---")
            st.success(f"💰 **Predicted Salary:** `${prediction:,.2f}`")

            # 🏅 Show experience level badge
            def get_experience_level(years):
                if years < 2:
                    return "🟢 Entry-Level"
                elif 2 <= years < 5:
                    return "🟡 Mid-Level"
                elif 5 <= years < 10:
                    return "🟠 Senior-Level"
                else:
                    return "🔴 Expert-Level"

            exp_level = get_experience_level(years_exp)
            st.markdown(f"**👤 Experience Level:** `{exp_level}`")

            # 📊 Salary bracket info
            if prediction < 50000:
                bracket = "🟢 Low"
            elif prediction < 100000:
                bracket = "🟡 Average"
            elif prediction < 150000:
                bracket = "🟠 High"
            else:
                bracket = "🔴 Very High"

            st.markdown(f"**📊 Salary Bracket:** `{bracket}`")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

elif option == "Batch Prediction":
    st.markdown("""
    <h1 style=' text-transform: uppercase;'>
        📂 Employee Salary Predictor
    </h1>
    <h2 style='text-align: center; text-transform: uppercase; font-weight: bold;'>
        Prediction For Batch Upload
    </h2>
""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:20px; font-weight:bold; margin-top:20px ; margin-botton: -10px;'>Upload CSV file with employee data</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"])

    st.markdown(
        "<p style='color: #d9534f; font-weight: bold; margin-top: 0px;'>⚠️ Ensure your CSV includes columns: Age, Gender, Education Level, Job Title, Years of Experience</p>", 
        unsafe_allow_html=True
    )
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if 'Salary' in batch_df.columns:
            batch_df = batch_df.drop(columns=['Salary'])

        required_cols = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
        if all(col in batch_df.columns for col in required_cols):
            try:
                batch_transformed = preprocessor.transform(batch_df[required_cols])
                predictions = model.predict(batch_transformed)
                predictions = np.round(predictions, 2)
                batch_df['Predicted Salary'] = predictions
                st.write("Predictions:")
                st.dataframe(batch_df)

                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Predictions as CSV", data=csv, file_name='predicted_salaries.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error(f"Uploaded file must contain the columns: {', '.join(required_cols)}")

elif option == "EDA & Visualization":
    st.markdown("""
    <h1 style='text-align: center; text-transform: uppercase;'>
      📊 Exploratory Data Analysis & Visualization
    </h1>
    <p style='text-align: left; font-size: 1.5em; font-weight: 600; margin-top: 20px; margin-bottom: 0px;'>
      Upload CSV for EDA (Optional)
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(label="", type=["csv"])  # empty label hides default text

    if uploaded_file is not None:
        eda_df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded dataset for EDA:")
        st.dataframe(eda_df.head())
    else:
        eda_df = default_df.copy()
        st.write("### Preview of default dataset:")
        st.dataframe(eda_df.head())

    num_cols = ['Age', 'Years of Experience', 'Salary']
    cat_cols = ['Gender', 'Education Level', 'Job Title']

    st.write("### Numeric Feature Distributions")
    for col in num_cols:
        if col in eda_df.columns:
            fig, ax = plt.subplots()
            sns.histplot(eda_df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)
            
        # Save plot to buffer for download
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
            label=f"Download Distribution Plot of {col}",
            data=buf,
            file_name=f'{col}_distribution.png',
            mime='image/png'
        )

    st.write("### Categorical Feature Counts with Filtering")

    for col in cat_cols:
        if col in eda_df.columns:
            # Special handling for Job Title: only top 10 by count
            if col == "Job Title":
                top_10_titles = eda_df[col].value_counts().nlargest(10).index.tolist()
                selected_values = st.multiselect(f"Select {col} (top 10 only)", options=top_10_titles, default=top_10_titles)
                filtered_df = eda_df[eda_df[col].isin(selected_values)]
            else:
                unique_vals = eda_df[col].dropna().unique()
                selected_values = st.multiselect(f"Select {col}", options=unique_vals, default=unique_vals)
                filtered_df = eda_df[eda_df[col].isin(selected_values)]

            fig, ax = plt.subplots(figsize=(12,6))
            sns.countplot(x=col, data=filtered_df, order=filtered_df[col].value_counts().index, ax=ax)
            plt.xticks(rotation=60, ha='right', fontsize=15)
            ax.set_title(f'Count of {col}')
            st.pyplot(fig)

             # Save plot to buffer for download
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
            label=f"Download Count Plot of {col}",
            data=buf,
            file_name=f'{col}_countplot.png',
            mime='image/png'
        )

    if all(col in eda_df.columns for col in num_cols):
        st.write("### Correlation Heatmap")
        corr = eda_df[num_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Save plot to buffer for download
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        st.download_button(
        label="Download Correlation Heatmap",
        data=buf,
        file_name='correlation_heatmap.png',
        mime='image/png'
    )

# Resume Parser Section        
elif option == "Resume Parser":
    st.markdown("""
    <h1 style='text-align: center; text-transform: uppercase;'>
      📄 Resume Parser & Salary Predictor
    </h1>
    <p style='text-align: left; font-size: 1.5em; font-weight: 600; margin-top: 20px; margin-bottom: 0px;'>
      Upload a Resume (PDF)
    </p>
    """, unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader(label="", type=["pdf"])

    if uploaded_pdf:
        try:
            text = ""
            with pdfplumber.open(uploaded_pdf) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            st.text_area("Extracted Resume Text (first 4000 chars):", text[:4000], height=300)

            # Try parsing experience
            try:
                years_exp = parse_experience(text)
            except Exception as e:
                years_exp = 0
                st.warning("⚠️ We encountered an error parsing your experience. Please enter it manually.")

            # Extract job title with fallback on error
            try:
                job_title = extract_job_title(text)
            except Exception as e:
                job_title = "Unknown"
                st.warning("⚠️ We encountered an error parsing your job title. Please enter it manually.")

            st.write(f"**Detected Job Title:** {job_title}")
            job_title = st.text_input("Please enter your current or desired job title, if not detected correctly:", value=job_title)

            if years_exp == 0:
                st.warning("⚠️ We couldn’t detect your years of experience from the resume. Please enter it manually below.")
            years_exp = st.number_input(
                "Enter your years of experience, if not detected correctly:",
                min_value=0.0, max_value=50.0, step=0.1, value=years_exp
            )

        except Exception as e:
            st.error(f"Failed to process the resume. Please check the file and try again.\nError: {e}")
            years_exp = st.number_input(
                "Enter your years of experience:",
                min_value=0.0, max_value=50.0, step=0.1, value=0.0
            )
            job_title = st.text_input("Enter your current or desired job title:", value="Unknown")

        # Additional inputs for prediction
        age = st.number_input("Enter your Age", min_value=18, max_value=70, value=30)
        gender = st.selectbox("Select Gender", options=["Male", "Female", "Other"], index=0)
        education = st.selectbox("Select Education Level", options=["High School", "Bachelor's", "Master's", "PhD", "Other"], index=1)

        if st.button("Predict Salary"):
            input_df = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Education Level": [education],
                "Job Title": [job_title],
                "Years of Experience": [years_exp]
            })

            try:
                input_transformed = preprocessor.transform(input_df)
                prediction = model.predict(input_transformed)[0]
                st.success(f"💰 Predicted Salary: {prediction:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.info("Please upload your resume PDF to begin parsing.")

elif option == "Salary Comparison Tool":
    st.markdown("""
    <h1 style='text-align: center; text-transform: uppercase; font-weight: bold; margin-bottom: 30px;'>
        📈 Salary Comparison Tool
    </h1>
    """, unsafe_allow_html=True)

    job_list = sorted(job_salary_benchmarks.keys())
    selected_job = st.selectbox("Select Job Title", job_list)

    if selected_job:
        benchmark_salary = job_salary_benchmarks.get(selected_job, None)
        if benchmark_salary:
            user_salary = st.number_input(
                f"Enter your current salary for **{selected_job}** (in USD):",
                min_value=0, step=1000, format="%d"
            )

            # Show benchmark salary info
            st.write(f"Average benchmark salary for **{selected_job}** is: 💰 ${benchmark_salary:,.2f} USD")

            if user_salary > 0:
                # Calculate the percentage difference
                percentage_difference = ((user_salary - benchmark_salary) / benchmark_salary) * 100

                # Create a DataFrame to compare user salary with benchmark salary
                salaries = pd.DataFrame({
                    'Category': ['Your Salary', 'Benchmark Salary'],
                    'Amount': [user_salary, benchmark_salary]
                })

                # Create a bar chart for salary comparison
                fig, ax = plt.subplots()
                bars = ax.bar(salaries['Category'], salaries['Amount'], color=['blue', 'orange'])
                ax.set_ylabel('Salary (USD)')
                ax.set_title(f'Salary Comparison for {selected_job}')

                # Annotate the bars with actual salary values
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'${height:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom')

                st.pyplot(fig)

                # Additional text-based comparison and percentage difference
                if user_salary > benchmark_salary:
                    st.success(f"🎉 Your salary of **${user_salary:,.2f}** is above the average benchmark by **{percentage_difference:,.2f}%**!")
                elif user_salary < benchmark_salary:
                    st.info(f"📉 Your salary of **${user_salary:,.2f}** is below the average benchmark by **{abs(percentage_difference):,.2f}%**.")
                else:
                    st.info(f"👌 Your salary of **${user_salary:,.2f}** matches the average benchmark.")
        else:
            st.warning("Salary data not available for the selected job.")

else:
    # About Section
    st.markdown("""
    <h1 style='text-align: center; text-transform: uppercase; font-weight: bold; margin-bottom: 20px;'>
        ℹ️ About The App
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
<style>
    .about-list {
        font-size: 18px;
        line-height: 1.6;
        color: #333333;
        padding-left: 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    .about-list li {
        margin-bottom: 12px;
    }
    .about-list li::before {
        content: "✅";
        margin-right: 10px;
        color: #2E8B57;
        font-weight: bold;
    }
</style>
<ul class="about-list">
    <li>Built with <strong>scikit-learn</strong> and <strong>Streamlit</strong> for seamless machine learning integration & interactive UI</li>
    <li>Predicts employee salary based on multiple relevant features: age, gender, education level, job title, and years of experience</li>
    <li>Supports both <em>individual</em> and <em>batch</em> salary predictions for flexibility and scalability</li>
    <li>Includes comprehensive Exploratory Data Analysis (EDA) tools and interactive data visualization to uncover insights</li>
    <li>Model trained on a large dataset of 6700+ employee records across diverse industries and roles</li>
    <li>User-friendly interface designed to assist HR professionals, recruiters, and job seekers in salary decisions</li>
    <li>Provides valuable insights for competitive salary benchmarking and informed negotiation strategies</li>
    <li>Continuous updates planned for improved accuracy, additional features, and enhanced user experience</li>
    <li>Handles data preprocessing automatically to ensure consistent and reliable predictions</li>
    <li>Supports uploading of resumes (PDF) for automatic parsing and salary prediction</li>
    <li>Batch upload feature enables processing of multiple employee records simultaneously, saving time</li>
    <li>Includes salary comparison tool to analyze and compare salaries across different job roles and experience levels</li>
    <li>Robust error handling ensures smooth user experience and clear feedback on input issues</li>
    <li>Built with modular and scalable codebase to facilitate future improvements and customization</li>
</ul>
""", unsafe_allow_html=True)
