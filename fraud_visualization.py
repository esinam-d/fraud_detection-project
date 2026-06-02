import pandas as pd  # For the data to be manipulated
import seaborn as sns    # For the data to be visualized
import matplotlib.pyplot as plt   # For the data to be visualized
import matplotlib.ticker as mticker    # For formatting y-axis labels with commas

# ─────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────
df = pd.read_csv("credcard_fraud.csv")

sns.set_theme(style="darkgrid", palette="muted") # Set a consistent theme for all plots
FRAUD_COLORS = {0: "#64E141", 1: "#E0392D"}   # blue = legit (no fraud), red = fraud

# ─────────────────────────────────────────────
#  Figure layout  (2 rows x 3 cols)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Fraud Detection — Exploratory Analysis", fontsize=18, fontweight="bold", y=1.01)

# ── Plot 1: Class balance (count) ────────────
ax = axes[0, 0]
counts = df["is_fraud"].value_counts().sort_index()
bars = ax.bar(["Non-Fraud (0)", "Fraud (1)"], counts.values,
              color=[FRAUD_COLORS[0], FRAUD_COLORS[1]], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Transaction Class Balance")
ax.set_ylabel("Number of Transactions")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# ── Plot 2: Class balance (pie) ───────────────
ax = axes[0, 1]
ax.pie(counts.values,
       labels=["Non-Fraud", "Fraud"],
       colors=[FRAUD_COLORS[0], FRAUD_COLORS[1]],
       autopct="%1.2f%%", startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("Fraud Proportion")

# ── Plot 3: Transaction amount distribution ───
ax = axes[0, 2]
if "amt" in df.columns:
    for label, grp in df.groupby("is_fraud"):
        sns.kdeplot(grp["amt"], ax=ax, label=f"{'Fraud' if label else 'Non-Fraud'}",
                    color=FRAUD_COLORS[label], fill=True, alpha=0.35)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(left=0)
else:
    ax.text(0.5, 0.5, "No 'amt' column found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Amount Distribution (N/A)")

# ── Plot 4: Amount box plot ───────────────────
ax = axes[1, 0]
if "amt" in df.columns:
    sns.boxplot(x="is_fraud", y="amt", data=df, ax=ax,
                palette={0: FRAUD_COLORS[0], 1: FRAUD_COLORS[1]})
    ax.set_title("Amount by Fraud Label (Boxplot)")
    ax.set_xlabel("is_fraud")
    ax.set_ylabel("Amount ($)")
    ax.set_xticklabels(["Non-Fraud", "Fraud"])
else:
    ax.set_visible(False)

# ── Plot 5: Hourly transaction count ─────────
ax = axes[1, 1]
if "trans_date_trans_time" in df.columns:
    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    hourly = df.groupby(["hour", "is_fraud"]).size().reset_index(name="count")
    for label, grp in hourly.groupby("is_fraud"):
        ax.plot(grp["hour"], grp["count"],
                label=f"{'Fraud' if label else 'Non-Fraud'}",
                color=FRAUD_COLORS[label], linewidth=2, marker="o", markersize=3)
    ax.set_title("Transactions by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Transaction Count")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
elif "Time" in df.columns:
    ax.hist(df[df["is_fraud"] == 0]["Time"], bins=48, alpha=0.5,
            color=FRAUD_COLORS[0], label="Non-Fraud", density=True)
    ax.hist(df[df["is_fraud"] == 1]["Time"], bins=48, alpha=0.7,
            color=FRAUD_COLORS[1], label="Fraud", density=True)
    ax.set_title("Transaction Time Distribution")
    ax.set_xlabel("Time (seconds)")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No time column found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Time Distribution (N/A)")

# ── Plot 6: Top fraud categories ─────────────
ax = axes[1, 2]
cat_col = next((c for c in ["category", "merchant_category", "type"] if c in df.columns), None)
if cat_col:
    fraud_by_cat = (df[df["is_fraud"] == 1][cat_col]
                    .value_counts().head(10))
    sns.barplot(x=fraud_by_cat.values, y=fraud_by_cat.index,
                ax=ax, palette="Reds_r")
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
