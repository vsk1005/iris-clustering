from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data

# Train K-Means
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X)

# Evaluation
score = silhouette_score(X, model.labels_)

print("Clusters:", model.n_clusters)
print("Silhouette Score:", score)

# Save model
joblib.dump(model, "iris_kmeans.pkl")