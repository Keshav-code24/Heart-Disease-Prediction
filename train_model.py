import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Load dataset
df = pd.read_csv("heart.csv")

# Features = columns 1 to 44
X = df.iloc[:, :-1]

# Target = last column (45)
y = df.iloc[:, -1]


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)


# Save KNN model
with open("knn_heart_model.pkl", "wb") as file:
    pickle.dump(model, file)


# Save scaler
with open("heart_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


# Save column names
with open("heart_columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)


print("3 files created successfully!")