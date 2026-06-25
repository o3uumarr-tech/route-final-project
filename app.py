
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

experience = st.number_input(
    "Experience Years",
    min_value=0,
    max_value=50,
    value=1
)

job_title = st.number_input("Job Title", 0, 20, 0)

education_level = st.number_input("Education Level", 0, 10, 0)

location = st.number_input("Location", 0, 20, 0)


predict = st.button("Predict Salary")








df=pd.read_csv(r"C:\Users\BASEL\OneDrive\Desktop\finalproj\job_salary_prediction_dataset.csv")


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

label_encoder = LabelEncoder()


for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

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




X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


from sklearn.linear_model import LinearRegression



linear.fit(X_train, y_train)

if predict:

    prediction = linear.predict([[
        experience,
        education_level,
        job_title,
        location
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
