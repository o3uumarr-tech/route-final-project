
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Job Salary Predictor",
    layout="wide"
)

st.title("Job Salary Prediction")

st.write("Enter employee information to predict salary")

st.sidebar.header("Employee Information")


df=pd.read_csv(r"C:\Users\BASEL\OneDrive\Desktop\finalproj\job_salary_prediction_dataset.csv")

job_titles = sorted(df["job_title"].unique())
locations = sorted(df["location"].unique())
education_levels = sorted(df["education_level"].unique())

experience = st.number_input(
    "Experience Years",
    min_value=0,
    max_value=50,
    value=1
)

job_title = st.selectbox(
    "Job Title",
    job_titles
)

education_level = st.selectbox(
    "Education Level",
    education_levels
)

location = st.selectbox(
    "Location",
    locations
)


predict = st.button("Predict Salary")






df.head()

df.info()

df.columns

df.shape

df.describe()

df.dtypes

df.duplicated().sum()

df.isnull().sum()

corr=df[['experience_years', 'salary','certifications','skills_count']].corr()

corr

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

df['experience_years'].value_counts().plot(kind='bar')

df['education_level'].value_counts().plot(kind='bar')

df['job_title'].value_counts()

df['location'].value_counts()

df.columns

from sklearn.preprocessing import LabelEncoder

job_encoder = LabelEncoder()
location_encoder = LabelEncoder()
education_encoder = LabelEncoder()

df["job_title"] = job_encoder.fit_transform(df["job_title"])
df["location"] = location_encoder.fit_transform(df["location"])
df["education_level"] = education_encoder.fit_transform(df["education_level"])

X = df[['experience_years',
        'education_level',
        'job_title',
        'location']]
y = df['salary']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(X_train, y_train)

## one hot encoder


from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.metrics import accuracy_score,mean_absolute_error





X_train.head()






from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


from sklearn.linear_model import LinearRegression



linear.fit(X_train, y_train)

if predict:

    job_title_encoded = job_encoder.transform([job_title])[0]
    location_encoded = location_encoder.transform([location])[0]
    education_encoded = education_encoder.transform([education_level])[0]

    prediction = linear.predict([[
        experience,
        education_encoded,
        job_title_encoded,
        location_encoded
    ]])[0]

    st.write("Predicted Salary:")
    st.success(f"${prediction:,.0f}")

y_pred = linear.predict(X_test)

r2_score(y_test, y_pred)


from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"accuracy_score ={round(r2_score(y_test, y_pred)*100, 2),4}%100%")

y_pred = linear.predict(X_test)

y_pred

df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
