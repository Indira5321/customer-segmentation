import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Training data
data = {
    "Annual Income": [15, 16, 17, 90, 95, 110, 70, 65, 40, 30],
    "Spending Score": [39, 81, 6, 77, 40, 15, 20, 80, 60, 50]
}

df = pd.DataFrame(data)

# Train KMeans model
model = KMeans(n_clusters=5, random_state=0)
model.fit(df)

# Save correct model file
joblib.dump(model, "Customer Segmentation.pkl")

print("Model saved as Customer Segmentation.pkl")
