# train_leukemia_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv(r"blood cancer prediction/dataset/biased_leukemia_dataset.csv")

# Step 2: Map target labels to numeric values
df['Leukemia_Status'] = df['Leukemia_Status'].map({'Positive': 1, 'Negative': 0})

# Step 3: Select relevant features
features = ['Age', 'WBC_Count', 'RBC_Count', 'Platelet_Count', 'Hemoglobin_Level', 'Bone_Marrow_Blasts']
X = df[features]
y = df['Leukemia_Status']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# Step 7: Save the trained model
joblib.dump(model, "blood_cancer_model.pkl")
print("🎉 Model saved as 'blood_cancer_model.pkl'")
