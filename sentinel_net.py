import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATASET
print("Loading datasets...")
# Using the paths you provided
train_df = pd.read_csv(r'/workspaces/SentinelNet-Network-Intrusion-Anomaly-Detection-System/Train_data.csv')
test_df = pd.read_csv(r'/workspaces/SentinelNet-Network-Intrusion-Anomaly-Detection-System/Balanced_Test.csv')

# 2. FEATURE ALIGNMENT
# Identify the target column (usually 'class' or 'label')
target_col = train_df.columns[-1] 
print(f"Detected target column as: '{target_col}'")

# Ensure test_df has a target column name that matches train_df
if target_col not in test_df.columns:
    test_df.rename(columns={test_df.columns[-1]: target_col}, inplace=True)

# 3. PREPROCESSING
# Identify categorical columns
cat_cols = train_df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    # We use transform only on test to maintain consistency
    if col in test_df.columns:
        # Handle unseen labels in test set by mapping them to a default
        test_df[col] = test_df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col].astype(str))

# Convert target to binary for Isolation Forest (Normal: 1, Anomaly: -1)
# Checking the most frequent value to define "Normal"
normal_val = train_df[target_col].mode()[0]
print(f"Mapping {normal_val} to 1 (Normal) and everything else to -1 (Anomaly)")

train_df[target_col] = np.where(train_df[target_col] == normal_val, 1, -1)
test_df[target_col] = np.where(test_df[target_col] == normal_val, 1, -1)

# Split Features and Target
X_train = train_df.drop([target_col], axis=1)
y_train = train_df[target_col]

# IMPORTANT: Force X_test to have the EXACT same columns as X_train
X_test = test_df.reindex(columns=X_train.columns, fill_value=0)
y_test = test_df[target_col]

# 4. TRAIN ISOLATION FOREST
print(f"Training on {X_train.shape[1]} features...")
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X_train)

# 5. PREDICT & EVALUATE
print("Predicting on test data...")
y_pred = model.predict(X_test)

print("\n--- SentinelNet Detection Report ---")
# Check if we have both classes in y_test to avoid errors in the report
unique_labels = np.unique(np.concatenate([y_test, y_pred]))
target_names = ['Anomaly', 'Normal'] if len(unique_labels) > 1 else None

print(classification_report(y_test, y_pred, target_names=target_names))

# 6. VISUALIZATION
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('SentinelNet: Intrusion Detection Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('detection_results.png')
print("\nSuccess! Results saved to 'detection_results.png'")