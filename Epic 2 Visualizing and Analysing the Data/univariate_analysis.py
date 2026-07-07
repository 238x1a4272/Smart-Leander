"""
Univariate Analysis
===================
Analyze individual features one at a time using histograms for numeric columns
and bar charts / pie charts for categorical columns.

Numeric features: ApplicantIncome, CoapplicantIncome, LoanAmount,
                  Loan_Amount_Term, Credit_History
Categorical features: Gender, Married, Dependents, Education,
                      Self_Employed, Property_Area, Loan_Status (target)

Generates plots saved to: Tasks/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("../Dataset/loan_prediction.csv")

print("=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------------------------
# Numeric features
# ---------------------------------------------------------------------------
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History"]

print("\n--- Numeric Features ---\n")
for col in numeric_cols:
    if col in df.columns:
        data = df[col].dropna()
        print(f"{col}:")
        print(f"  Count      : {len(data)}")
        print(f"  Mean       : {data.mean():.2f}")
        print(f"  Median     : {data.median():.2f}")
        print(f"  Std Dev    : {data.std():.2f}")
        print(f"  Min        : {data.min():.2f}")
        print(f"  Max        : {data.max():.2f}")
        print(f"  Skew       : {data.skew():.2f}")
        print()

# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------
cat_cols = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area", "Loan_Status"]

print("\n--- Categorical Features ---\n")
for col in cat_cols:
    if col in df.columns:
        counts = df[col].value_counts()
        props = df[col].value_counts(normalize=True) * 100
        print(f"{col}:")
        for cat, count, pct in zip(counts.index, counts.values, props.values):
            print(f"  {cat:15s} : {count:4d}  ({pct:5.1f}%)")
        print(f"  Missing: {df[col].isnull().sum()}")
        print()

# ---------------------------------------------------------------------------
# Insight summary
# ---------------------------------------------------------------------------
print("\n--- Key Insights ---")
print("1. Target is imbalanced: ~69% Approved (Y), ~31% Rejected (N)")
print("2. Most applicants are Male, Married, Graduate, not Self-Employed")
print("3. Most applicants have 0 Dependents and are from Semiurban areas")
print("4. ApplicantIncome is right-skewed with a long tail (high earners)")
print("5. Most applicants have Credit_History = 1 (good credit)")
print("6. Missing values exist in Gender (13), Married (3), Dependents (15),")
print("   Self_Employed (32), LoanAmount (22), Loan_Amount_Term (14),")
print("   and Credit_History (50)")

# ===========================================================================
# VISUALIZATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Plot 1: Numeric features — histograms
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if col in df.columns:
        ax = axes[i]
        data = df[col].dropna()
        if col == "Credit_History":
            # Credit_History is discrete (0/1) — bar chart
            counts = data.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values,
                   color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.5)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            for j, v in enumerate(counts.values):
                ax.text(j, v + 5, str(v), ha="center", fontweight="bold")
        else:
            ax.hist(data, bins=25, color="#3498db", edgecolor="white",
                    linewidth=1.2, alpha=0.85)
            ax.axvline(data.median(), color="#e74c3c", linestyle="--",
                       linewidth=2, label=f"Median={data.median():.0f}")
            ax.axvline(data.mean(), color="#2ecc71", linestyle="--",
                       linewidth=2, label=f"Mean={data.mean():.0f}")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=10)
# Hide unused subplot
for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])
fig.suptitle("Univariate Analysis — Numeric Features", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "univariate_numeric.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[PLOT] Saved: {os.path.join(PLOTS_DIR, 'univariate_numeric.png')}")

# ---------------------------------------------------------------------------
# Plot 2: Categorical features — bar charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()
colors = sns.color_palette("viridis", 8)
for i, col in enumerate(cat_cols):
    if col in df.columns:
        ax = axes[i]
        counts = df[col].value_counts()
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=colors[:len(counts)], edgecolor="white", linewidth=1.5)
        ax.set_title(f"Distribution of {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                    f"{int(height)}", ha="center", va="bottom", fontweight="bold", fontsize=10)
# Hide unused subplots
for i in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[i])
fig.suptitle("Univariate Analysis — Categorical Features", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "univariate_categorical.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved: {os.path.join(PLOTS_DIR, 'univariate_categorical.png')}")

# ---------------------------------------------------------------------------
# Plot 3: Target variable — pie chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["Loan_Status"].value_counts()
colors_target = ["#2ecc71" if k == "Y" else "#e74c3c" for k in counts.index]
wedges, texts, autotexts = ax.pie(
    counts.values, labels=counts.index, autopct="%1.1f%%",
    colors=colors_target, startangle=90, explode=(0.05, 0.05),
    shadow=False, textprops={"fontsize": 14, "fontweight": "bold"}
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
ax.set_title("Target Variable: Loan_Status (Approved vs Rejected)",
             fontsize=14, fontweight="bold", pad=20)
fig.savefig(os.path.join(PLOTS_DIR, "univariate_target.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved: {os.path.join(PLOTS_DIR, 'univariate_target.png')}")
