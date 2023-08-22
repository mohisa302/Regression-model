# import libraris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Load data
df = pd.read_excel('/content/Data 280 (2).xlsx', dtype=np.float32)

# Extract feature columns
X = df[['Temperature', 'Volume', 'size']]
y = df['z']

# Split data into train, validate, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Get feature names
feature_names = poly.get_feature_names_out(input_features=X.columns)
feature_names = [f.replace(' ', '*') for f in feature_names]

# Ridge regression
ridge = Ridge(alpha=1E-8)
ridge.fit(X_train_poly, y_train)

# Get coefficients and intercept
coefficients = ridge.coef_
intercept = ridge.intercept_

# Generate regression equation string
equation = 'Y = '
for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
    if coef != 0:
        if coef > 0 and i > 0:
            equation += f' + {coef:.4f} * {feature}'
        elif coef > 0 and i == 0:
            equation += f'{coef:.4f} * {feature}'
        elif coef < 0:
            equation += f' - {abs(coef):.4f} * {feature}'

# Add the intercept term to the equation
if intercept >= 0:
    equation += f' + {intercept:.4f}'
else:
    equation += f' - {abs(intercept):.4f}'

# Print regression equation
print("Regression Equation:")
print(equation)

# Predict on test data
y_pred = ridge.predict(X_test_poly)

# Calculate R2 score for manually calculated results
r2_score_manual = r2_score(y_test, y_pred)

# Print manually calculated values, real data, and model predictions for test data
print()
print("Manually Calculated\t\tReal Data\t\tModel Predictions")
for i in range(len(X_test_poly)):
    manual_calc = sum(X_test_poly[i] * coefficients) + intercept
    print(f"{manual_calc:.4f}\t\t\t{y_test.iloc[i]}\t\t\t{y_pred[i]:.4f}")

# Create a DataFrame to store the results
results_df = pd.DataFrame({'Manually Calculated': [sum(X_test_poly[i] * coefficients) + intercept for i in range(len(X_test_poly))],
                           'Real Data': y_test,
                           'Model Predictions': y_pred})

# Save the results to an Excel file
results_df.to_excel('results.xlsx', index=False)

# Print R2 score for the manually calculated results
print()
print(f"R2 Score (Manual Calculation): {r2_score_manual:.4f}")
