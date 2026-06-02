import pandas as pd
import csv
import seaborn as sb
import matplotlib.pyplot as plt


# # 1. Read the CSV file into a DataFrame
df = pd.read_csv('credcard_fraud.csv')

# # 2. Plot a histogram of a specific column
# df['is_fraud'].hist(bins=2)

# # 3. Add labels and a title (using Matplotlib functions)
# plt.title('Number of Fraud Transactions')
# plt.xlabel('is_fraud')
# plt.ylabel('Number of Trans.')

# # 4. Display the plot
# plt.show()

# Count plot for fraud vs non-fraud
plt.figure(figsize=(5,7))
sb.countplot(x='is_fraud', data=df)
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('is_fraud')
plt.ylabel('Number of Transactions')
plt.show()

