import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("predictive_maintenance_dataset.csv")

# Features (correct column names)
X = df[['Temperature',
        'Vibration',
        'Pressure',
        'Humidity',
        'RPM',
        'Voltage',
        'Current',
        'Machine_Age']]

# Target column
y = df['Failure']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully and saved as model.pkl")