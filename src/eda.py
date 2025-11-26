"""
EDA script for the heart disease dataset.

Run:
    python -m src.eda
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .data_utils import load_heart_data


def basic_overview(df: pd.DataFrame) -> None:
    print("=== Basic info ===")
    print(df.info())
    print("\n=== Head ===")
    print(df.head())
    print("\n=== Describe (numeric) ===")
    print(df.describe())
    print("\n=== Describe (categorical) ===")
    print(df.describe(include="object"))

    print("\n=== Missing values per column ===")
    print(df.isna().sum())


def target_distribution(df: pd.DataFrame) -> None:
    print("\n=== Target distribution (HeartDisease) ===")
    print(df["HeartDisease"].value_counts())
    print("\n=== Target distribution (relative) ===")
    print(df["HeartDisease"].value_counts(normalize=True))

    sns.countplot(x="HeartDisease", data=df)
    plt.title("Target Distribution – HeartDisease")
    plt.tight_layout()
    plt.savefig("reports/target_distribution.png")
    plt.close()


def numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(exclude="object").columns

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"reports/dist_{col}.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig("reports/corr_heatmap.png")
    plt.close()


def categorical_vs_target(df: pd.DataFrame) -> None:
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        ct = pd.crosstab(df[col], df["HeartDisease"], normalize="index")
        print(f"\n=== {col} vs HeartDisease (row-normalized) ===")
        print(ct)

        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, hue="HeartDisease", data=df)
        plt.title(f"{col} vs HeartDisease")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"reports/{col}_vs_HeartDisease.png")
        plt.close()


def main():
    df = load_heart_data()

    # Ensure reports directory exists
    import os
    os.makedirs("reports", exist_ok=True)

    basic_overview(df)
    target_distribution(df)
    numeric_distributions(df)
    categorical_vs_target(df)

    print("\nEDA plots saved in the 'reports/' directory.")


if __name__ == "__main__":
    main()
