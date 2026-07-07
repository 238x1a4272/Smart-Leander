"""
Import and Read Dataset
=======================
Import required libraries and read the loan_prediction dataset.
Performs basic data inspection: shape, data types, statistical summary,
missing values, duplicates, and target distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------
# Read dataset
# ---------------------------------------------------------------------------
DATA_PATH = "../Dataset/loan_prediction.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("IMPORT AND READ DATASET")
print("=" * 60)

# Shape
print(f"\nShape: {df.shape}")

# First few rows
print("\nFirst 5 rows:")
print(df.head())

# Data types
print("\nData types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical summary:")
print(df.describe(include='all').T)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicates
print(f"\nDuplicates: {df.duplicated().sum()}")

# Target distribution
print("\nLoan_Status distribution:")
print(df['Loan_Status'].value_counts())

print("\n" + "=" * 60)
print("Columns:", list(df.columns))
print("=" * 60)
