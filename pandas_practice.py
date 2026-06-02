import pandas as pd 

#Creating small sample data set
data= {
    "Amount": [100,250,500,75],
    "Location": ["US", "US", "UK", "NG"],
    "Fraud": [0,0,1,1]
}

#Convert dictionary to DataFrame
df= pd.DataFrame(data)

print(df)

#Show first 2 rows
print("\nFirst 2 rows:")
print(df.head(2))

#Count fraud vs non fraud
print("\nFraud counts")
print (df["Fraud"].value_counts())

#Calculate perecentage of fraud
total =len(df)
fraud=(df["Fraud"].sum())
percentage= (fraud/total)*100
print(f"\nPerecentage of fraud transactions: {percentage}%")