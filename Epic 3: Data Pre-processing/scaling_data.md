# Task 12: Scaling the Data

## Project Name

**Smart Lender – Loan Eligibility Prediction System**

---

# Objective

Scale numerical features so that all values lie within a comparable range, improving machine learning model performance.

---

# Introduction

Features such as Applicant Income and Loan Amount have widely different ranges. Scaling prevents larger value ranges (e.g., thousands of dollars in income) from dominating smaller value ranges (e.g., credit history binary digits or loan terms), ensuring balanced feature weights in optimization.

---

# Feature Scaling Workflow

```mermaid
graph TD
    Start([1. Start: Numerical Features with Varied Scales]) --> Identify["2. Identify Columns <br> (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)"]
    
    Identify --> Choose{"3. Select Scaling Algorithm"}
    
    Choose -->|Standard Normalization <br> Mean=0, Std=1| SS["StandardScaler <br> z = (x - u) / s"]
    Choose -->|Range Bound Scaling <br> Range = [0, 1]| MMS["MinMaxScaler <br> x_scaled = (x - min) / (max - min)"]
    
    SS & MMS --> FitTrans["4. Fit and Transform Features <br> (scaler.fit_transform)"]
    FitTrans --> Scaled["5. Scaled Numerical Matrix"]
    Scaled --> Train["6. Model Training Phase <br> (Faster Convergence & Stable Weights)"]
    Train --> End([7. End: Optimized Model Output])
    
    style Start fill:#dfd,stroke:#333
    style End fill:#dfd,stroke:#333
    style SS fill:#ffd,stroke:#333
    style MMS fill:#ffd,stroke:#333
```

---

# Methods Used

## 1. StandardScaler (Standardization)

Transforms features to follow a standard normal distribution where:
* **Mean (\(\mu\))** = 0
* **Standard Deviation (\(\sigma\))** = 1

### Python Code Example:
```python
from sklearn.preprocessing import StandardScaler

# Select features to scale
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform features
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

---

## 2. MinMaxScaler (Normalization)

Scales and bounds values between a specific range (typically **0 and 1**):
* **Min** = 0
* **Max** = 1

### Python Code Example:
```python
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
min_max_scaler = MinMaxScaler()

# Fit and transform features
df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])
```

---

# Features Scaled

The following numerical columns are transformed:
* `ApplicantIncome`
* `CoapplicantIncome`
* `LoanAmount`
* `Loan_Amount_Term`

---

# Expected Output

* **Normalized numerical values:** Features adjusted to comparable standard scales.
* **Improved convergence:** Optimization algorithms (e.g. gradient descent) reach global minima faster.
* **Better prediction accuracy:** Equalized weight distributions improve validation stability.

---

# Benefits

* **Faster training:** Minimizes numerical calculations during backpropagation and gradient steps.
* **Improved model stability:** Eliminates extreme outlier dominance.
* **Better KNN/Distance performance:** Essential for distance-based algorithms (like KNN and SVM) to calculate geometric separation fairly.

---

# Conclusion

Feature scaling standardizes numerical values and significantly improves machine learning model efficiency.
