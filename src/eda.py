from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_eda(df: pd.DataFrame):

    print("\n===== DATA OVERVIEW =====")
    print(df.info())

    print("\nShape:", df.shape)

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["no_show"].value_counts())

    print("\n===== TARGET PERCENTAGE =====")
    print(df["no_show"].value_counts(normalize=True) * 100)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum().sort_values(ascending=False))

    sns.set_style("whitegrid")

    # 1. No-show distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="no_show")
    plt.title("Target Distribution (No Show)")
    plt.savefig(
        OUTPUT_DIR / "target_distribution.png",
        bbox_inches="tight"
    )
    plt.close()

    # 2. Days wait distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df["days_wait"], bins=30, kde=True)
    plt.title("Days Wait Distribution")
    plt.savefig(
        OUTPUT_DIR / "days_wait_distribution.png",
        bbox_inches="tight"
    )
    plt.close()

    # 3. Weekday vs No-show
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df,
        x="appointment_weekday",
        hue="no_show"
    )

    plt.title("No Show by Weekday")
    plt.xticks(rotation=45)

    plt.savefig(
        OUTPUT_DIR / "weekday_no_show.png",
        bbox_inches="tight"
    )
    plt.close()

    # 4. Age vs No-show
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df,
        x="no_show",
        y="Age"
    )

    plt.title("Age vs No Show")

    plt.savefig(
        OUTPUT_DIR / "age_vs_no_show.png",
        bbox_inches="tight"
    )
    plt.close()

    print("\nEDA completed.")
    print("Charts saved in outputs/")