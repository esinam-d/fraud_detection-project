import pandas as pd  # For the data to be manipulated
import seaborn as sns    # For the data to be visualized
import matplotlib.pyplot as plt   # For the data to be visualized
import matplotlib.ticker as mticker    # For formatting y-axis labels with commas

# ─────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────
df = pd.read_csv("credcard_fraud.csv") #This will load the dataset into the Data Frame (df).

sns.set_theme(style="darkgrid", palette="muted") # Set a consistent theme for all plots
FRAUD_COLOURS = {0: "#64E141", 1: "#E0392D"}   # blue = legit (no fraud), red = fraud

# ─────────────────────────────────────────────
#  Figure layout  (2 rows x 3 cols)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("FRAUD DETECTION GRAPHICS", fontsize=16, fontweight="bold", y=1.01)

# Plot 1: Class balance (count)
ax = axes[0, 0] 
counts = df["is_fraud"].value_counts().sort_index()
bars = ax.bar(["Non-Fraud (0)", "Fraud (1)"], counts.values,
              color=[FRAUD_COLOURS[0], FRAUD_COLOURS[1]], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Transaction Class Balance")
ax.set_ylabel("Number of Transactions")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Plot 2: Class balance (pie chart)
# A pie chart to show the proportion of fraud vs non-fraud transactions. It uses the same counts as the bar chart but visualizes them as slices of a pie, with labels and percentages for clarity.
ax = axes[0, 1]
ax.pie(counts.values,
       labels=["Non-Fraud", "Fraud"], # x-axis= Non-Fraud, y-axis= Fraud
       colors=[FRAUD_COLOURS[0], FRAUD_COLOURS[1]],
       autopct="%1.2f%%", startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("Fraud Proportion")

# Plot 3: Transaction amount distribution 
ax = axes[0, 2]
if "amt" in df.columns:
    for label, grp in df.groupby("is_fraud"):
        sns.kdeplot(grp["amt"], ax=ax, label=f"{'Fraud' if label else 'Non-Fraud'}",
                    color=FRAUD_COLOURS[label], fill=True, alpha=0.35)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(left=0)
else:
    ax.text(0.5, 0.5, "No 'amt' column found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Amount Distribution (N/A)") # If the 'amt' column is missing, show a message instead of the plot

# Plot 4: Amount box plot 
ax = axes[1, 0]
if "amt" in df.columns:
    df["is_fraud_str"] = df["is_fraud"].astype(str)
    sns.boxplot(x="is_fraud_str", y="amt", data=df, ax=ax,
                hue="is_fraud_str",
                palette={"0": FRAUD_COLOURS[0], "1": FRAUD_COLOURS[1]},
                order=["0", "1"],
                legend=False)
    ax.set_title("Amount by Fraud Label (Boxplot)")
    ax.set_xlabel("is_fraud")
    ax.set_ylabel("Amount ($)")
    ax.set_xticklabels(["Non-Fraud", "Fraud"])
else:
    ax.set_visible(False)

# Plot 5: Hourly transaction count 
ax = axes[1, 1]
# This plot shows how the number of transactions varies by hour of the day, comparing fraud and non-fraud transactions. It extracts the hour from a datetime column (if available) and plots line graphs for both classes to identify any patterns in transaction timing.
if "trans_date_trans_time" in df.columns:
    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    hourly = df.groupby(["hour", "is_fraud"]).size().reset_index(name="count")
    for label, grp in hourly.groupby("is_fraud"):
        ax.plot(grp["hour"], grp["count"],
                label=f"{'Fraud' if label else 'Non-Fraud'}",
                color=FRAUD_COLOURS[label], linewidth=2, marker="o", markersize=3)
    ax.set_title("Transactions by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Transaction Count")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()

# If the dataset doesn't have a date-time column, it falls back to plotting the distribution of a "Time" column (if available) or shows a message if neither is present.
elif "Time" in df.columns:
    ax.hist(df[df["is_fraud"] == 0]["Time"], bins=48, alpha=0.5,
            color=FRAUD_COLOURS[0], label="Non-Fraud", density=True)
    ax.hist(df[df["is_fraud"] == 1]["Time"], bins=48, alpha=0.7,
            color=FRAUD_COLOURS[1], label="Fraud", density=True)
    ax.set_title("Transaction Time Distribution")
    ax.set_xlabel("Time (seconds)")
    ax.legend()

else:
    ax.text(0.5, 0.5, "No time column found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Time Distribution (N/A)")

# Plot 6: Top fraud categories 
ax = axes[1, 2]
cat_col = next((c for c in ["category", "merchant_category", "type"] if c in df.columns), None)
if cat_col:
    fraud_by_cat = (df[df["is_fraud"] == 1][cat_col]
                    .value_counts().head(10))
    sns.barplot(x=fraud_by_cat.values, y=fraud_by_cat.index,
                ax=ax, color=FRAUD_COLOURS[1])
    ax.set_title(f"Top 10 Fraud Categories\n(by '{cat_col}')")
    ax.set_xlabel("Fraud Count")
    ax.set_ylabel("")
else:
    ax.text(0.5, 0.5, "No category column found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Fraud by Category (N/A)")

# ─────────────────────────────────────────────
#  Save & show
# ─────────────────────────────────────────────
plt.tight_layout() 
plt.savefig("fraud_analysis.png", dpi=150, bbox_inches="tight")
print("Saved → fraud_analysis.png")
plt.show()
