import pandas as pd # For data manipulation
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For splitting data into train/test sets
from sklearn.preprocessing import LabelEncoder, StandardScaler # For encoding categorical variables and scaling features
from sklearn.linear_model import LogisticRegression    # For the logistic regression model
from sklearn.ensemble import RandomForestClassifier  # For the random forest model

# For evaluation metrics and visualization
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt #To plot the results
import warnings #This will ignore warnings from the libraries. Keeps the output clean.
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  1. Load data
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("credcard_fraud.csv") #Load the data into Data Frame (df).

# ─────────────────────────────────────────────
#  2. Preprocess data
# ─────────────────────────────────────────────
# Drop columns that leak identity or are non-informative
DROP_COLS = ["trans_num", "first", "last", "street", "dob",
             "unix_time", "merchant", "cc_num"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Parse datetime features if available
if "trans_date_trans_time" in df.columns:
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]      = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"]     = dt.dt.month
    df.drop(columns=["trans_date_trans_time"], inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "is_fraud" in cat_cols:
    cat_cols.remove("is_fraud")

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ─────────────────────────────────────────────
#  3. Train / test split
# ─────────────────────────────────────────────
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print dataset sizes and fraud rate in training set
print(f"Train size : {len(X_train):,}   |   Test size : {len(X_test):,}")
print(f"Fraud rate in train : {y_train.mean()*100:.3f}%")

# ─────────────────────────────────────────────
#  4. Handle class imbalance with SMOTE
# ─────────────────────────────────────────────
print("\nApplying SMOTE to balance training set...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Resampled train size : {len(X_train_res):,}  "
      f"(fraud={y_train_res.sum():,} / non-fraud={(y_train_res==0).sum():,})")

# ─────────────────────────────────────────────
#  5. Scale features
# ─────────────────────────────────────────────
scaler = StandardScaler() # Scale features for Logistic Regression 
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
#  6. Train the Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), 
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

# Train each model, make predictions, and evaluate using ROC-AUC and classification report. Store results for visualization.
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Random Forest doesn't need scaled data, LR does
    Xt = X_train_sc if name == "Logistic Regression" else X_train_res
    Xe = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(Xt, y_train_res)
    y_pred  = model.predict(Xe)
    y_proba = model.predict_proba(Xe)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    results[name] = {"model": model, "y_pred": y_pred,
                     "y_proba": y_proba, "roc_auc": roc_auc,
                     "X_test": Xe}
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Non-Fraud", "Fraud"]))

# ─────────────────────────────────────────────
#  7. Visualize results
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fraud Detection Model Evaluation", fontsize=14, fontweight="bold")

# Confusion matrices
for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Fraud", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\n(ROC-AUC: {res['roc_auc']:.4f})")

# ROC curves 
ax = axes[2]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
ax.set_title("ROC Curves")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("fraud_model_results.png", dpi=150, bbox_inches="tight")
print("\nSaved → fraud_model_results.png")
plt.show()

# ─────────────────────────────────────────────
#  8. Feature importance (Random Forest)
# ─────────────────────────────────────────────
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()

plt.figure(figsize=(8, 6))
top15.plot(kind="barh", color="#4C9BE8", edgecolor="white")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("fraud_feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved → fraud_feature_importance.png")
plt.show()

print("\nDone! Best model by ROC-AUC:",
      max(results, key=lambda k: results[k]["roc_auc"]))
