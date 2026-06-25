import pandas as pd
import matplotlib.pyplot as plt
import os

print(" Classifier Comparison:")
print("_" * 60)


dt_results = {"Model": "Decision Tree","Accuracy": 0.5714,"Precision": 0.7143,"Recall": 0.5556,"F1-Score": 0.6250}
nb_results = {"Model": "Naive Bayes","Accuracy": 0.5000,"Precision": 0.4250,"Recall": 0.4333,"F2-Score": 0.4269}

#comparasion table 
comparison_df = pd.DataFrame([dt_results, nb_results])

print("\nClassifier Comparison Results:")
print(comparison_df.to_string(index=False))

PLOTS_DIR = "."
os.makedirs(PLOTS_DIR, exist_ok=True)

comparison_df.to_csv(os.path.join(PLOTS_DIR, "classifier_comparison_results.csv"),index=False,encoding="utf-8-sig")


plt.figure(figsize=(6, 4))
plt.bar(comparison_df["Model"], comparison_df["Accuracy"])
plt.title("Decision Tree vs Naive Bayes - Accuracy")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "comparison_accuracy.png"), dpi=200)
plt.show()
plt.close()

#plot f-score comparion 
plt.figure(figsize=(6, 4))
plt.bar(comparison_df["Model"], comparison_df["F1-Score"])
plt.title("Decision Tree vs Naive Bayes - F1-Score")
plt.xlabel("Classifier")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "comparison_f1_score.png"), dpi=200)
plt.show()
plt.close()

print("\nComparison files saved successfully.")
print("Generated files:")
print(" - classifier_comparison_results.csv")
print(" - comparison_accuracy.png")
print(" - comparison_f1_score.png")