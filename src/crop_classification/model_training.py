import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the trained model

# load the dataset
df = pd.read_csv("final_training_data.csv")

# separate features and labels
X = df.drop(columns=["field_id", "label"])
y = df["label"]

# split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,  # prevents overfitting
    random_state=42
)

# train the model
rf_model.fit(X_train, y_train)

# make predictions
y_pred = rf_model.predict(X_test)

# evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Model saved as 'random_forest_model.pkl'")
