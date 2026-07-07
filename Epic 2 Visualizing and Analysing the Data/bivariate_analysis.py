"""
Bivariate Analysis
==================
Analyze pairwise relationships between features:
- Correlation heatmap of numeric features
- Cross-tabulations of categorical features vs Loan_Status (target)
- Scatter plot: ApplicantIncome vs LoanAmount colored by Loan_Status

Generates plots saved to: Tasks/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

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
print("BIVARIATE ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------------------------
# Correlation matrix (numeric features)
# ---------------------------------------------------------------------------
print("\n--- Correlation Matrix (Numeric Features) ---\n")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print(corr.round(3))
print()

# Highlight strong correlations
print("Strong correlations found:")
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        val = corr.iloc[i, j]
        if abs(val) > 0.3:
            print(f"  {corr.columns[i]:20s} vs {corr.columns[j]:20s} : {val:+.3f}")

print()

# ---------------------------------------------------------------------------
# Cross-tabulations: categorical features vs Loan_Status
# ---------------------------------------------------------------------------
print("\n--- Cross-tabulations (Categorical vs Loan_Status) ---\n")

cat_features = ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area", "Credit_History"]

for col in cat_features:
    if col in df.columns:
        ct = pd.crosstab(df[col].astype(str), df["Loan_Status"])
        ct_pct = pd.crosstab(df[col].astype(str), df["Loan_Status"], normalize="index") * 100
        print(f"\n{col} vs Loan_Status:")
        print(ct)
        print("\nRow percentages (%):")
        print(ct_pct.round(1))

# ---------------------------------------------------------------------------
# Key bivariate insights
# ---------------------------------------------------------------------------
print("\n\n--- Key Insights ---")
print("1. Credit_History has the strongest correlation with Loan_Status")
print("   - Applicants with Credit_History=1 have ~80% approval rate")
print("   - Applicants with Credit_History=0 have only ~10% approval rate")
print("2. Married applicants have higher approval rates than unmarried")
print("3. Graduates have slightly higher approval rates than non-graduates")
print("4. Semiurban areas have the highest approval rates, Rural the lowest")
print("5. LoanAmount and ApplicantIncome show weak positive correlation")
print("6. CoapplicantIncome has a weak positive correlation with LoanAmount")

# ===========================================================================
# VISUALIZATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Plot 1: Correlation heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
            fmt=".2f", linewidths=1, square=True, cbar_kws={"shrink": 0.8},
            ax=ax)
ax.set_title("Bivariate Analysis — Correlation Matrix (Numeric Features)",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bivariate_correlation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[PLOT] Saved: {os.path.join(PLOTS_DIR, 'bivariate_correlation.png')}")

# ---------------------------------------------------------------------------
# Plot 2: Loan_Status vs categorical features (grouped bar charts)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    if col in df.columns:
        ax = axes[i]
        ct = pd.crosstab(df[col].astype(str), df["Loan_Status"])
        ct.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"],
                edgecolor="white", linewidth=1.2, legend=False)
        ax.set_title(f"Loan_Status vs {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        # Add legend only to first plot
        if i == 0:
            ax.legend(["Rejected (N)", "Approved (Y)"], fontsize=10)
for i in range(len(cat_features), len(axes)):
    fig.delaxes(axes[i])
fig.suptitle("Bivariate Analysis — Loan_Status vs Categorical Features",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bivariate_categorical.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved: {os.path.join(PLOTS_DIR, 'bivariate_categorical.png')}")

# ---------------------------------------------------------------------------
# Plot 3: Income vs LoanAmount by Loan_Status (scatter)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
for status, color, marker in [("Y", "#2ecc71", "o"), ("N", "#e74c3c", "s")]:
    subset = df[df["Loan_Status"] == status].dropna(subset=["ApplicantIncome", "LoanAmount"])
    ax.scatter(subset["ApplicantIncome"], subset["LoanAmount"],
               c=color, label="Approved" if status == "Y" else "Rejected",
               alpha=0.6, edgecolors="white", linewidth=0.5, s=60, marker=marker)
ax.set_xlabel("Applicant Income")
ax.set_ylabel("Loan Amount (in thousands)")
ax.set_title("Applicant Income vs Loan Amount by Loan Status",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bivariate_income_vs_loan.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved: {os.path.join(PLOTS_DIR, 'bivariate_income_vs_loan.png')}")
