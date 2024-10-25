import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
salary_data = pd.read_csv('Salary_Data.csv')

# Clean the data by dropping rows with missing values
salary_data = salary_data.dropna(subset=['Years of Experience', 'Salary'])

# Select relevant columns
X = salary_data[['Years of Experience']]
y = salary_data['Salary']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# App Title
st.title("Salary Prediction Based on Years of Experience")

# User input for years of experience
years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 1.0, step=0.1)

# Predict salary based on input
predicted_salary = model.predict(np.array([[years_of_experience]]))[0]

# Display the prediction
st.write(f"Predicted Salary for {years_of_experience} years of experience: ${predicted_salary:,.2f}")

# Plotting the data and regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience')
plt.legend()

# Display the plot
st.pyplot(plt)

# Calculate R² score on both the training and testing sets
r2_score_train = model.score(X_train, y_train)
r2_score_test = model.score(X_test, y_test)

# Display model performance
st.write(f"Model R² Score on Training Set: {r2_score_train:.2f}")
st.write(f"Model R² Score on Test Set: {r2_score_test:.2f}")
