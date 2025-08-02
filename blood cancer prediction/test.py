import joblib
import pandas as pd
model = joblib.load(r"blood cancer prediction/models/blood_cancer_model.pkl")

sample = pd.DataFrame([[52, 2698, 5.36, 262493, 12.2, 72]],
                      columns=['Age', 'WBC_Count', 'RBC_Count', 'Platelet_Count', 'Hemoglobin_Level', 'Bone_Marrow_Blasts'])
prediction = model.predict(sample)
print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
