"""
Multivariate Analysis
=====================
Analyze interactions among multiple features simultaneously:
- Pairplot of key numeric features colored by Loan_Status
- Box plots of numeric features grouped by Loan_Status
- Violin plots of ApplicantIncome and LoanAmount by Loan_Status

Generates plots saved to: Tasks/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

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
print("MULTIVARIATE ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------------------------
# Numeric summary grouped by Loan_Status
# ---------------------------------------------------------------------------
print("\n--- Numeric Features by Loan_Status ---\n")
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term"]

for col in numeric_cols:
    if col in df.columns:
        print(f"\n{col} by Loan_Status:")
        grouped = df.groupby("Loan_Status")[col].describe()
        print(grouped.round(2).to_string())

# ---------------------------------------------------------------------------
# Statistical tests / feature interactions
# ---------------------------------------------------------------------------
print("\n\n--- Feature Interaction Analysis ---\n")

# Income-based features
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
print("TotalIncome (Applicant + Coapplicant) by Loan_Status:")
print(df.groupby("Loan_Status")["TotalIncome"].describe().round(2))

# Loan-to-income ratio
df_valid = df.dropna(subset=["LoanAmount", "ApplicantIncome"]).copy()
df_valid["LoanToIncomeRatio"] = df_valid["LoanAmount"] * 1000 / df_valid["ApplicantIncome"]
print("\nLoan-to-Income Ratio by Loan_Status:")
print(df_valid.groupby("Loan_Status")["LoanToIncomeRatio"].describe().round(2))

# Cross-tab of Property_Area and Credit_History vs Loan_Status
print("\nCredit_History × Property_Area × Loan_Status Rate:")
ct = df.groupby(["Credit_History", "Property_Area"])["Loan_Status"] \
       .apply(lambda x: (x == "Y").mean() * 100).round(1)
print(ct)

# ---------------------------------------------------------------------------
# Key multivariate insights
# ---------------------------------------------------------------------------
print("\n\n--- Key Insights ---")
print("1. Credit_History dominates as the strongest predictor — regardless")
print("   of other features, applicants with Credit_History=1 are far more")
print("   likely to be approved.")
print("2. The loan-to-income ratio matters: rejected applicants tend to have")
print("   higher ratios (larger loans relative to income).")
print("3. Property_Area interacts with Credit_History:")
print("   - Semiurban + good credit -> highest approval rate")
print("   - Rural + no credit history -> lowest approval rate")
print("4. Married Graduates with good credit from Semiurban areas have the")
print("   highest approval likelihood.")
print("5. Applicants with high income AND low loan amounts are almost always")
print("   approved if they also have good credit history.")

# ===========================================================================
# VISUALIZATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Plot 1: Pairplot of numeric features colored by Loan_Status
# ---------------------------------------------------------------------------
# Prepare data
plot_df = df.drop(columns=["Loan_ID"], errors="ignore").copy()
plot_df["Loan_Status_Encoded"] = plot_df["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})
pair_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
pair_df = plot_df[pair_features + ["Loan_Status_Encoded"]].dropna()

g = sns.PairGrid(pair_df, hue="Loan_Status_Encoded",
                 palette={"Approved": "#2ecc71", "Rejected": "#e74c3c"},
                 diag_sharey=False, corner=False)
g.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor="white", linewidth=0.5)
g.map_lower(sns.kdeplot, alpha=0.3, levels=4, fill=True)
g.map_diag(sns.histplot, alpha=0.7, edgecolor="white", linewidth=0.8, bins=20)
g.add_legend(title="Loan Status", fontsize=11)
g.fig.suptitle("Multivariate Analysis — Pairplot of Numeric Features",
               fontsize=16, fontweight="bold", y=1.02)
g.fig.savefig(os.path.join(PLOTS_DIR, "multivariate_pairplot.png"), dpi=150, bbox_inches="tight")
plt.close(g.fig)
print(f"\n[PLOT] Saved: {os.path.join(PLOTS_DIR, 'multivariate_pairplot.png')}")

# ---------------------------------------------------------------------------
# Plot 2: Box plots grouped by Loan_Status
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df_box = df.dropna(subset=[col, "Loan_Status"])
    bp = ax.boxplot(
        [df_box[df_box["Loan_Status"] == s][col].values for s in ["Y", "N"]],
        labels=["Approved (Y)", "Rejected (N)"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "white", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_title(f"{col} by Loan Status", fontweight="bold")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
fig.suptitle("Multivariate Analysis — Box Plots Grouped by Loan Status",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "multivariate_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved: {os.path.join(PLOTS_DIR, 'multivariate_boxplots.png')}")

# ---------------------------------------------------------------------------
# Plot 3: Violin plots for key distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, col in enumerate(["ApplicantIncome", "LoanAmount"]):
    ax = axes[i]
    df_violin = df.dropna(subset=[col, "Loan_Status"])
    parts = ax.violinplot(
        [df_violin[df_violin["Loan_Status"] == s][col].values for s in ["Y", "N"]],
        positions=[1, 2], showmeans=True, showmedians=True, widths=0.7
    )
    parts["bodies"][0].set_facecolor("#2ecc71")
    parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_facecolor("#e74c3c")
    parts["bodies"][1].set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Approved (Y)", "Rejected (N)"])
    ax.set_ylabel(col)
    ax.set_title(f"{col} Distribution by Loan Status (Violin Plot)", fontweight="bold")
    ax.grid(True, alpha=0.3)
fig.suptitle("Multivariate Analysis — Violin Plots",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "multivariate_violin.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved: {os.path.join(PLOTS_DIR, 'multivariate_violin.png')}")
