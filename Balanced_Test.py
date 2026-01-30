import pandas as pd

# 1. Load your training data (which has the labels)
train_df = pd.read_csv(r'/workspaces/SentinelNet-Network-Intrusion-Anomaly-Detection-System/Train_data.csv')

# 2. Separate Normal and Anomaly rows
normal_traffic = train_df[train_df['class'] == 'normal']
anomaly_traffic = train_df[train_df['class'] == 'anomaly']

# 3. Take a sample from each to create a balanced test set (e.g., 2000 each)
# This ensures your "Support" will be 2000 for both in the final report
test_normal = normal_traffic.sample(n=2000, random_state=42)
test_anomaly = anomaly_traffic.sample(n=2000, random_state=42)

# 4. Combine them
balanced_test = pd.concat([test_normal, test_anomaly])

# 5. Save this as your new test file
balanced_test.to_csv('Balanced_Test.csv', index=False)

print("Success! Created 'Balanced_Test.csv' with 2000 Normal and 2000 Anomaly rows.")