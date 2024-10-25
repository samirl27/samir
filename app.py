import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# Load the dataset
salary_data = pd.read_csv('Salary_Data.csv')

# Data cleaning
salary_data = salary_data.dropna(subset=['Years of Experience', 'Salary'])

# Feature and target selection
X = salary_data[['Years of Experience']]
y = salary_data['Salary']

# Experiment with Polynomial Features for non-linear relationships
degree = st.selectbox("Choose the polynomial degree for model fitting:", [1, 2, 3], index=1)
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display title
st.title("Enhanced Salary Prediction Model")

# Input slider for user experience
years_of_experience = st.slider('Years of Experience', 0, 40, 1)
input_features = poly.transform(np.array([[years_of_experience]]))
predicted_salary = model.predict(input_features)[0]

# Display prediction
st.write(f"Predicted Salary for {years_of_experience} years of experience: ${predicted_salary:,.2f}")

# Cross-validation for accuracy estimate
cv_scores = cross_val_score(model, X_poly, y, cv=5)
avg_cv_score = np.mean(cv_scores)

# Model performance on test data
r2_test_score = r2_score(y_test, y_pred)

# Display the model performance
st.write(f"Model R² Score on test data: {r2_test_score:.2f}")
st.write(f"Average cross-validation R² Score: {avg_cv_score:.2f}")

# Plot data and polynomial regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(np.sort(X, axis=0), model.predict(np.sort(X_poly, axis=0)), color='red', label=f'{degree}-degree Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience')
plt.legend()

# Display the plot
st.pyplot(plt)
