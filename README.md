# Energy-Efficiency
Problem - Evaluating the impact of building design and construction on its heating and cooling demands and, consequently, its energy efficiency. Using Advanced Statistical Methods.

# Research Question:
Can we create a Machine Learning model to evaluate the heating and cooling demands of a residential building using only its engineering design variables?

# Machine Learning Models:
## Ridge Regression
Serves as a baseline to capture linear relationships

Performed 10-fold CV over λ∈ [0,50]; recorded optimal λ(0.25) and corresponding to the minimum MSE

## Lasso Regression (Ridge + Lasso)

Select interaction terms using Lasso regularization (combined with Ridge)

Performed 10-fold CV over α∈ [0,1] and λ∈ [0,50]; recorded the optimal (α,λ) and choosing (α= 1,λ= 0.002) which has minimum MSE

## Spline Regression with Lasso Regularization:

Fitting a spline regression with the mgcv library

Regularize using a L1 regularization

# KNN:
Motivation - ”Similar” residential structures may have similar energy load.

Run 10-fold CV for k ∈ [1,50], using the best k(5) with the lowest MSE.

Random Forest: As proposed by the Initial Researchers

# Results:

<img width="611" height="422" alt="Screenshot 2025-11-04 at 8 56 30 PM" src="https://github.com/user-attachments/assets/39b65edb-e4b2-4e30-bdac-77cfce98334a" />

<img width="610" height="346" alt="Screenshot 2025-11-04 at 8 56 51 PM" src="https://github.com/user-attachments/assets/e7c1fdf0-dc4f-418c-8dc3-1826b66acf18" />
