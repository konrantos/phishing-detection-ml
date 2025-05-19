# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
import seaborn as sns  # For confusion matrix visualization
from scipy.io import arff

# Load dataset
file_path = "Training Dataset.arff"
raw_data = arff.loadarff(file_path)
data = pd.DataFrame(raw_data[0])

# Initial plot of data distribution before preprocessing
plt.figure(figsize=(6, 4))
value_counts = data["Result"].value_counts()
value_counts.plot(kind="bar", color=["skyblue", "salmon"])
for index, value in enumerate(value_counts):
    plt.text(index, value + 10, str(value), ha="center", fontsize=12)
plt.xlabel("Category", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)
plt.title("Class Distribution")
plt.xticks([0, 1], ["Phishing", "Legitimate"], rotation=0)
plt.show()

# Data preprocessing
# Convert byte columns to integer
for column in data.columns:
    data[column] = data[column].astype(str).astype(int)

# Split features and labels
X = data.drop(columns=["Result"])  # All except 'Result'
y = data["Result"]

# -------------------------------- Decision Tree Model --------------------------------

# Find best max_leaf_nodes
best_max_leaf_nodes = None
best_accuracy = 0

for max_leaf in range(200, 500, 2):
    dt_temp = DecisionTreeClassifier(random_state=1, max_leaf_nodes=max_leaf)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_scores = cross_val_score(dt_temp, X, y, cv=kf)
    accuracy_temp = np.mean(cv_scores)
    if accuracy_temp > best_accuracy:
        best_accuracy = accuracy_temp
        best_max_leaf_nodes = max_leaf

# Final Decision Tree model with best parameter
dt = DecisionTreeClassifier(random_state=1, max_leaf_nodes=best_max_leaf_nodes)
kf = KFold(n_splits=10, shuffle=True, random_state=1)
print(
    f"Decision Tree:\nBest max_leaf_nodes: {best_max_leaf_nodes} with mean accuracy {best_accuracy * 100:.3f}% (10-fold CV)"
)

# Confusion matrix using cross-validation
y_pred_all = cross_val_predict(dt, X, y, cv=kf)
cm_dt_cv = confusion_matrix(y, y_pred_all)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_dt_cv,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"],
    linewidths=1,
    linecolor="black",
    annot_kws={"size": 16},
)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.title("Confusion Matrix — Decision Tree (10-Fold CV)")
plt.show()

# Metrics for Decision Tree
accuracy_dt = accuracy_score(y, y_pred_all)
recall_dt = recall_score(y, y_pred_all)
f1_dt = f1_score(y, y_pred_all)

print(f"Decision Tree Accuracy: {accuracy_dt * 100:.3f}%")
print(f"Decision Tree Recall: {recall_dt * 100:.3f}%")
print(f"Decision Tree F1 Score: {f1_dt * 100:.3f}%")
print("-" * 150)

# -------------------------------- k-Nearest Neighbors Model --------------------------------

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different k values (1 to 15)
k_values = range(1, 15)
mean_accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf)
    mean_accuracy = np.mean(cv_scores)
    mean_accuracies.append(mean_accuracy)

# Plot accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_accuracies, marker="o")
plt.xlabel("k (Number of Neighbors)", fontsize=14)
plt.ylabel("Mean Accuracy", fontsize=14)
plt.title("k-NN Accuracy vs. k (10-Fold CV)", fontsize=16)
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Find best k
best_k = k_values[np.argmax(mean_accuracies)]
print(
    f"k-NN:\nBest k: {best_k} with mean accuracy {mean_accuracies[np.argmax(mean_accuracies)] * 100:.3f}% (10-fold CV)"
)

# Final k-NN model with best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
y_pred_all_knn = cross_val_predict(knn_best, X_scaled, y, cv=kf)
cm_knn_cv = confusion_matrix(y, y_pred_all_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_knn_cv,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"],
    linewidths=1,
    linecolor="black",
    annot_kws={"size": 12},
)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.title(f"Confusion Matrix — k-NN (k={best_k}, 10-Fold CV)")
plt.show()

# Metrics for k-NN
accuracy_knn = accuracy_score(y, y_pred_all_knn)
recall_knn = recall_score(y, y_pred_all_knn)
f1_knn = f1_score(y, y_pred_all_knn)

print(f"k-NN Accuracy: {accuracy_knn * 100:.3f}%")
print(f"k-NN Recall: {recall_knn * 100:.3f}%")
print(f"k-NN F1 Score: {f1_knn * 100:.3f}%")
