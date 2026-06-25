import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, accuracy_score,precision_recall_fscore_support,ConfusionMatrixDisplay)

#initial
print("BIRZEIT UNIVERSITY - Neurofibromatosis Classification")
print("_" * 60)

#data loading 
def load_and_clean_dataset():
    try:
        df = pd.read_csv("../data/uci_dataset.csv")
        df = df.dropna(how="all")
        # Drop unnamed index column if present
        if df.columns[0] == "" or "Unnamed" in df.columns[0]:
            df = df.drop(df.columns[0], axis=1)

        df.columns = df.columns.str.strip()

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None
    except Exception as e:
        print("Unexpected error while loading dataset:", e)
        return None


df = load_and_clean_dataset()

if df is None:
    exit()

# Verify columns loaded correctly
print("\n[Columns Loaded]")
print(list(df.columns))

target_column = "Case Type"

if target_column not in df.columns:
    print(f"Error: Target column '{target_column}' not found.")
    print("Available columns:", list(df.columns))
    exit()

#dataset description 
print("\ndataset description")
print("_" * 60)
print(f"Total Instances: {len(df)}")
print(f"Number of Features: {df.shape[1] - 1}")
print(f"Target Variable: {target_column}")
print(f"Class Distribution: {dict(df[target_column].value_counts())}")

print("\nAll Features:")
feature_cols = [c for c in df.columns if c != target_column]
for f in feature_cols:
    print(f"  - {f}")

print("\nMissing Values per Column")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("No missing values found")
else:
    for col, count in missing.items():
        print(f"  {col}: {count} missing")

X = df.drop(target_column, axis=1)
y = df[target_column]


# Continuous columns: (appropriate for age values)
# Binary columns : (appropriate for 0/1 flags)
continuous_cols = ["Age of Mother", "Age of Father", "Age at First Diagnosis"]
continuous_cols = [c for c in continuous_cols if c in X.columns]
binary_cols     = [c for c in X.columns if c not in continuous_cols]

print("\nImputation Strategy")
print(f"Median (continuous): {continuous_cols}")
print(f"Most frequent (binary): {binary_cols}")

ct = ColumnTransformer([("median_imputer", SimpleImputer(strategy="median"),continuous_cols),("freq_imputer",SimpleImputer(strategy="most_frequent"), binary_cols),], remainder="passthrough")

X_imputed = ct.fit_transform(X)

# Rebuild column name order to match ColumnTransformer output
imputed_columns = continuous_cols + binary_cols
X_imputed = pd.DataFrame(X_imputed, columns=imputed_columns)

print("\n[After Imputation]")
print(f"  Missing values remaining: {pd.DataFrame(X_imputed).isnull().sum().sum()}")

#train test 80,20
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y,test_size=0.20,random_state=42,stratify=y)

print("\nDataset Split:80% Train,20% Test")
print(f"Training Instances: {X_train.shape[0]}")
print(f"Testing Instances: {X_test.shape[0]}")

print("\nModel Training")
print("Training Gaussian Naive Bayes...")
print("GaussianNB chosen because dataset contains continuous age features")

model = GaussianNB()
model.fit(X_train, y_train)
print("Model trained successfully.")

y_pred = model.predict(X_test)

acc=accuracy_score(y_test, y_pred)
prec, rec, f1m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

print("test set 20%:")
print("_" * 60)

results_df = pd.DataFrame([{"Model": "Gaussian Naive Bayes","Accuracy" : round(acc,  4),"Precision": round(prec, 4),"Recall"   : round(rec,  4),"F1-Score" : round(f1m,  4),}])
print(results_df.to_string(index=False))

print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=["Sporadic (0)", "Familial (1)"], zero_division=0))

#plots
PLOTS_DIR = "partC_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot1: Class Distribution
plt.figure(figsize=(6, 4))
class_counts = df[target_column].value_counts().sort_index()
plt.bar(["Sporadic (0)", "Familial (1)"], class_counts.values, color="steelblue")
plt.title("Neurofibromatosis — Class Distribution")
plt.xlabel("Case Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=200)
plt.show()
plt.close()

# Plot2:Performance Metrics Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Precision", "Recall", "F1-Score"], [acc, prec, rec, f1m], color="steelblue")
plt.title("Gaussian Naive Bayes — Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
for i, v in enumerate([acc, prec, rec, f1m]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "metrics_bar.png"), dpi=200)
plt.show()
plt.close()

# Plot3: Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,display_labels=["Sporadic (0)", "Familial (1)"],cmap="Blues",values_format="d")
plt.title("Gaussian Naive Bayes — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=200)
plt.show()
plt.close()

# Plot4: Missing Values Before Imputation
if len(missing) > 0:
    plt.figure(figsize=(7, 4))
    missing.plot(kind="bar", color="salmon")
    plt.title("Missing Values per Column (Before Imputation)")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "missing_values.png"), dpi=200)
    plt.show()
    plt.close()

#save result
results_df.to_csv( os.path.join(PLOTS_DIR, "final_results_table.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame({ "Actual"   : y_test.values, "Predicted": y_pred}).to_csv(  os.path.join(PLOTS_DIR, "predictions.csv"),index=False, encoding="utf-8-sig")

print(f"\nAll plots and results saved to: ./{PLOTS_DIR}/")
print("Generated files:")
print("  - class_distribution.png")
print("  - metrics_bar.png")
print("  - confusion_matrix.png")
print("  - missing_values.png  (if missing values existed)")
print("  - final_results_table.csv")
print("  - predictions.csv")
print("\nAll tasks completed successfully.")