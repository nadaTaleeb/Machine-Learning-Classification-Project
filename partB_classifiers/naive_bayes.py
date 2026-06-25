import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# initialization 
print("BIRZEIT UNIVERSITY - PlayTennis Naive Bayes Classifier")
print("_" * 60)
print("Starting system:")

#load data
def load_and_clean_dataset():
    try:
        df = pd.read_csv("../data/playtennis.csv")
        df = df.dropna(how="all")
        expected_columns = ["Outlook", "Temp", "Humidity", "Wind", "PlayTennis"]
        for col in expected_columns:
            if col not in df.columns:
                print(f"Error: Missing required column")
                return None
        for col in expected_columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

    except Exception as e:
        print(f"error loading dataset: {e}")
        return None


#encoding
def encode_dataset(df):
    X = df[["Outlook", "Temp", "Humidity", "Wind"]]
    y = df["PlayTennis"]

    X_encoded = X.copy()
    feature_encoders = {}

    for col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        feature_encoders[col] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    return X_encoded, y_encoded, feature_encoders, target_encoder


#Display the probability tables
def print_probability_tables(model, feature_encoders, target_encoder):
    print("\nlearned naive bayes probability tables")
    print("_" * 60)

    print("\nClass Prior Probabilities:")
    for i, class_name in enumerate(target_encoder.classes_):
        prob = np.exp(model.class_log_prior_[i])
        print(f"P(PlayTennis = {class_name}) = {prob:.4f}")

    print("\nConditional Probabilities:")
    for feature_index, feature_name in enumerate(feature_encoders.keys()):
        print(f"\nFeature: {feature_name}\n")  #current feature

        categories = feature_encoders[feature_name].classes_

        for class_index, class_name in enumerate(target_encoder.classes_): # Loop through each class (Yes / No)
            print(f"  Given PlayTennis = {class_name}:\n")
            probs = np.exp(model.feature_log_prob_[feature_index][class_index])
            for category, prob in zip(categories, probs):
                print(f"    P({feature_name} = {category} | PlayTennis = {class_name}) = {prob:.4f}")


#model training 
print("\nPreparing Naive Bayes implementation...")

df = load_and_clean_dataset()

if df is None:
    exit()

print("\n[Dataset Analysis]")
print(f"Total Samples: {len(df)}")
print(f"Class Counts: {dict(df['PlayTennis'].value_counts())}")

X_encoded, y_encoded, feature_encoders, target_encoder = encode_dataset(df)

print("\n[Feature Engineering]")
print(f"Feature Count: {X_encoded.shape[1]}")
print(f"Encoded Features: {list(X_encoded.columns)}")


print("\nTraining Naive Bayes using Leave-One-Out Cross Validation...")

nb_final = CategoricalNB(alpha=1.0)

loo = LeaveOneOut()

y_pred = cross_val_predict(nb_final,X_encoded,y_encoded,cv=loo)

accuracy = accuracy_score(y_encoded, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_encoded,y_pred,average="macro",zero_division=0)


print("\nfinal evaluation using leave-one-out")
print("_" * 90)

results_summary = [{"Model": "Naive Bayes","Accuracy": accuracy,"Precision": precision,"Recall": recall,"F1-Score": f1}]

comparison_df = pd.DataFrame(results_summary)

print(comparison_df.to_string(index=False))

print("\nClassification Report:")
print(classification_report(y_encoded,y_pred,target_names=target_encoder.classes_,zero_division=0))

# train final model on all data for probability tables and sample prediction
nb_final.fit(X_encoded, y_encoded)

print_probability_tables(nb_final, feature_encoders, target_encoder)


print("\nsample  prediction")
print("_" * 60)

sample = X_encoded.iloc[[0]]
prediction = nb_final.predict(sample)
predicted_label = target_encoder.inverse_transform(prediction)[0]

print("Encoded sample:")
print(sample)
print(f"Predicted PlayTennis = {predicted_label}")


plots = "partB_classifiers"
os.makedirs(plots, exist_ok=True)

#plot class distribution 
plt.figure(figsize=(6, 4))
df["PlayTennis"].value_counts().plot(kind="bar")
plt.title("PlayTennis Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plots, "class_distribution.png"), dpi=200)
plt.show()
plt.close()

#plot Naive Bayes metrics
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Precision", "Recall", "F1-Score"],[accuracy, precision, recall, f1])
plt.title("Naive Bayes Performance Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(plots, "naive_bayes_metrics.png"), dpi=200)
plt.show()
plt.close()

#plot confusion matrix 
ConfusionMatrixDisplay.from_predictions(y_encoded,y_pred,display_labels=target_encoder.classes_,values_format="d")

plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(plots, "naive_bayes_confusion_matrix.png"), dpi=200)
plt.show()
plt.close()

#Save final results 
comparison_df.to_csv(os.path.join(plots, "naive_bayes_results_table.csv"),index=False,encoding="utf-8-sig")

print(f"\n* Plots saved to: ./{plots}/")
print("Generated files:\n")
print(" - class_distribution.png")
print(" - naive_bayes_metrics.png")
print(" - naive_bayes_confusion_matrix.png")
print(" - naive_bayes_results_table.csv")