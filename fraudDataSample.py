import pandas as pd

# Read the CSV fraud data file
df= pd.read_csv("credcard_fraud.csv")

# Show first 5 rows
print("="*50)
print("DATASET PREVIEW:")
print("="*50)

# Check the shape of the data
print("\nNumber of rows and columns:")
print(df.shape)

# Count fraud vs non-fraud
fraud_counts = df["is_fraud"].value_counts()
print("\nFraud counts:")
print(fraud_counts)

# Calculate percentage of fraud transactions
total = len(df)
fraud = df["is_fraud"].sum()  # '1' indicates fraud
percentage = (fraud / total) * 100
print(f"\nPercentage of fraud transactions: {percentage:.4f}%")