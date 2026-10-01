# Week 6 - Integrative Capstone Project
# Wine Analysis: Full Data Science Pipeline
# Prepared by: JOYDEEP DAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, adjusted_rand_score
)

SEED = 42
np.random.seed(SEED)

# =========================================================
# 1. Data acquisition
# =========================================================
wine = load_wine(as_frame=True)

X = wine.data.copy()
y = wine.target.copy()

df = X.copy()
df["target"] = y
df["target_name"] = y.map(dict(enumerate(wine.target_names)))

print("Dataset shape:", X.shape)
print("\nFeature names:")
print(list(X.columns))
print("\nClass distribution:")
print(y.value_counts().sort_index())
print("\nMissing values:", X.isnull().sum().sum())

df.to_csv("Wine_Capstone_Week6.csv", index=False)

# =========================================================
# 2. Exploratory Data Analysis
# =========================================================
print("\nDescriptive statistics:")
print(X.describe().T)

plt.figure(figsize=(7, 4))
y.value_counts().sort_index().plot(kind="bar")
plt.title("Wine Dataset Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 7))
plt.imshow(X.corr().values, aspect="auto")
plt.title("Feature Correlation Matrix")
plt.xticks(range(len(X.columns)), X.columns, rotation=90, fontsize=7)
plt.yticks(range(len(X.columns)), X.columns, fontsize=7)
plt.colorbar()
plt.tight_layout()
plt.show()

# =========================================================
# 3. Supervised learning - Random Forest
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        max_features="sqrt",
        class_weight="balanced"
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

cv_results = cross_validate(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted"
    ]
)

print("\nFive-Fold Cross-Validation:")
for metric in [
    "test_accuracy",
    "test_precision_weighted",
    "test_recall_weighted",
    "test_f1_weighted"
]:
    print(
        metric,
        ":",
        round(cv_results[metric].mean(), 4),
        "+/-",
        round(cv_results[metric].std(), 4)
    )

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

print("\nTest Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nTest Precision:",
      precision_score(y_test, y_pred, average="weighted"))

print("\nTest Recall:",
      recall_score(y_test, y_pred, average="weighted"))

print("\nTest F1:",
      f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Feature importance
rf_model = rf_pipeline.named_steps["model"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Feature Importances:")
print(importance.head(10))

plt.figure(figsize=(8, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# =========================================================
# 4. Unsupervised learning - K-Means
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []

for k in range(2, 8):
    kmeans = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

best_k = list(range(2, 8))[
    int(np.argmax(silhouette_scores))
]

print("\nSelected number of clusters:", best_k)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=SEED,
    n_init=20
)

cluster_labels = final_kmeans.fit_predict(X_scaled)

print(
    "Silhouette Score:",
    silhouette_score(X_scaled, cluster_labels)
)

print(
    "Adjusted Rand Index against known labels:",
    adjusted_rand_score(y, cluster_labels)
)

# Elbow plot
plt.figure(figsize=(7, 4))
plt.plot(
    range(2, 8),
    inertias,
    marker="o"
)
plt.title("K-Means Elbow Analysis")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

# Silhouette plot
plt.figure(figsize=(7, 4))
plt.plot(
    range(2, 8),
    silhouette_scores,
    marker="o"
)
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# =========================================================
# 5. PCA visualization of clusters
# =========================================================
pca = PCA(
    n_components=2,
    random_state=SEED
)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))

for cluster in range(best_k):

    mask = cluster_labels == cluster

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster}"
    )

plt.title("K-Means Clusters in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 6. Cluster profiling
# =========================================================
clustered_data = X.copy()
clustered_data["cluster"] = cluster_labels

cluster_profile = clustered_data.groupby(
    "cluster"
).mean()

print("\nCluster Profile:")
print(cluster_profile.round(3))

cluster_profile.to_csv(
    "Wine_Cluster_Profile_Week6.csv"
)

print("\nCapstone project completed successfully.")
