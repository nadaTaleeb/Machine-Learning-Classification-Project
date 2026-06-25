# Decision Tree classification on PlayTennis dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

# load and clean dataset
try:
    data = pd.read_csv("data/playtennis.csv")
    print("Dataset loaded successfully.")

except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

except Exception as e:
    print("Unexpected error while loading dataset:", e)
    exit()

# check if dataset is empty
if data.empty:
    print("Error: Dataset is empty.")
    exit()

# remove completely empty lines
data = data.dropna(how="all")

# required columns for PlayTennis dataset
expected_columns = ["Outlook", "Temp", "Humidity", "Wind", "PlayTennis"]

for col in expected_columns:
    if col not in data.columns:
        print(f"Error: Missing required column: {col}")
        exit()

# skip rows that have missing values
missing_values = data[expected_columns].isnull().sum().sum()

if missing_values > 0:
    print("Warning:", missing_values, "missing value(s) found. Incomplete rows will be skipped.")
    data = data.dropna(subset=expected_columns)

# check again after cleaning
if data.empty:
    print("Error: No valid rows left after cleaning the dataset.")
    exit()


# keep only needed columns
data = data[expected_columns]

print("\nFirst rows of cleaned dataset:")
print(data.head())

# convert text values to numbers because ML models work with numbers
try:
    encoders = {}

    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    print("\nEncoding completed successfully.")

except Exception as e:
    print("Encoding error:", e)
    exit()

# split inputs and target
X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]


# build Decision Tree model using entropy
model = DecisionTreeClassifier(criterion="entropy", random_state=42)

# evaluate using Leave-One-Out because dataset is small
loo = LeaveOneOut()

# predictions using Leave-One-Out
y_pred = cross_val_predict(model, X, y, cv=loo)

fig, ax = plt.subplots(figsize=(5,5))
ConfusionMatrixDisplay.from_predictions(y,y_pred,display_labels=encoders["PlayTennis"].classes_,cmap="Blues",ax=ax)
plt.title("Confusion Matrix (LOO) - Decision Tree")
plt.savefig("confusion_matrix_dt.png", dpi=300, bbox_inches="tight")
plt.show()

# evaluation metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

print("\nEvaluation Metrics")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-score :", round(f1 * 100, 2), "%")

# train final model on all data for visualization
model.fit(X, y)

# draw and save decision tree
plt.figure(figsize=(16, 8))
plot_tree(model,feature_names=X.columns,class_names=["No", "Yes"],filled=True,rounded=True,fontsize=10)
plt.title("Decision Tree - PlayTennis Dataset")

output_file = "tree_visualization.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
print("\nDecision Tree visualization saved as:", output_file)
