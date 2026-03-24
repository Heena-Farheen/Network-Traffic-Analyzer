# =============================================================================
#  UNSW-NB15 SUPERVISED IDS — FINAL COMPLETE VERSION
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# =============================================================================
# USER CONFIGURATION
# =============================================================================

TRAIN_PATH = r"C:\Users\HP\OneDrive\Desktop\Network_Traffic\UNSW_NB15_training-set.parquet"
TEST_PATH  = r"C:\Users\HP\OneDrive\Desktop\Network_Traffic\UNSW_NB15_testing-set.parquet"
TARGET_COLUMN = "label"   # Change to "attack_cat" if needed
RANDOM_STATE = 42

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print("\n Loading Dataset...")

train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)

print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)

print("\nColumns in Training Data:")
print(train_df.columns)

# =============================================================================
# STEP 2 — PREPROCESSING
# =============================================================================

print("\n Preprocessing...")

train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

# Fill missing values
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

# Separate features and labels
y_train = train_df[TARGET_COLUMN]
y_test  = test_df[TARGET_COLUMN]

X_train = train_df.drop(columns=[TARGET_COLUMN, "attack_cat"])
X_test  = test_df.drop(columns=[TARGET_COLUMN, "attack_cat"])

# Detect numeric and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

import sklearn
_ohe_kwargs = {"handle_unknown": "ignore"}
if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2):
    _ohe_kwargs["sparse_output"] = False
else:
    _ohe_kwargs["sparse"] = False

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(**_ohe_kwargs), cat_cols)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test  = preprocessor.transform(X_test)

# Apply SMOTE only to training data
if pd.Series(y_train).value_counts(normalize=True).min() < 0.40:
    print("Applying SMOTE...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# =============================================================================
# STEP 3 — MODEL TRAINING
# =============================================================================

print("\n Training Models...")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),

    "Random Forest": GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators":[100], "max_depth":[None]},
        cv=2, scoring="f1_weighted", n_jobs=-1
    ),

   

    "KNN": GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors":[5]},
        cv=2, scoring="f1_weighted", n_jobs=-1
    )
}

trained_models = {}

print("\n" + "="*50)
print(" STARTING MODEL OPTIMIZATION PIPELINE")
print("="*50)

for name, model in tqdm(models.items(), desc="Overall Models", unit="model"):

    print(f"\n Optimizing {name}...")

    trained_models[name] = model.fit(X_train, y_train)

    print(f" {name} Optimization Complete!")

# =============================================================================
# STEP 4 — EVALUATION
# =============================================================================

print("\n Evaluating Models...")

results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    try:
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except:
        roc_auc = 0

    results.append([name, acc*100, prec, rec, f1, roc_auc])

    print("\n==============================")
    print("Model:", name)
    print("Accuracy:", round(acc*100,2))
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results,
                          columns=["Model","Accuracy","Precision","Recall","F1","ROC-AUC"])

results_df.sort_values("Accuracy", ascending=False, inplace=True)

print("\n Accuracy Comparison Table:\n")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)

# =============================================================================
# STEP 5 — VISUALIZATION
# =============================================================================

sns.set_style("whitegrid")

# Accuracy Bar Chart
plt.figure(figsize=(10,6))
sns.barplot(x="Accuracy", y="Model", data=results_df)
plt.title("Model Accuracy Comparison")
plt.savefig("accuracy_comparison.png")
plt.show()

# Confusion Matrices
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"cm_{name}.png")
    plt.show()

# ROC Curves (Binary only)
if len(np.unique(y_test)) == 2:
    plt.figure(figsize=(8,6))
    for name, model in trained_models.items():
        try:
            y_prob = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        except:
            continue

    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig("roc_curves.png")
    plt.show()

print("\n IDS Pipeline Completed Successfully!")
print("Outputs saved:")
print("- model_comparison_results.csv")
print("- accuracy_comparison.png")
print("- cm_*.png")
print("- roc_curves.png (if binary)")

# =============================================================================
# STEP 6 — THRESHOLD TUNING (RECALL PRIORITY)
# =============================================================================

print("\n Performing Threshold Optimization (Maximizing Recall)...")

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

if hasattr(best_model, "predict_proba") and len(np.unique(y_test)) == 2:
    y_probs = best_model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 50)
    best_recall = 0
    best_threshold = 0.5

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        recall = recall_score(y_test, y_pred_thresh)

        if recall > best_recall:
            best_recall = recall
            best_threshold = t

    print(f"Best Threshold for Recall: {best_threshold:.3f}")
    print(f"Best Recall Achieved: {best_recall:.4f}")

# =============================================================================
# STEP 7 — STACKING ENSEMBLE
# =============================================================================

from sklearn.ensemble import StackingClassifier

print("\n Training Stacking Ensemble...")

stack_model = StackingClassifier(
    estimators=[
        ("rf", trained_models["Random Forest"]),
        ("gb", trained_models["Gradient Boosting"]),
        ("lr", trained_models["Logistic Regression"])
    ],
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

y_stack_pred = stack_model.predict(X_test)

print("\nStacking Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_stack_pred)*100, 2))
print("Recall:", round(recall_score(y_test, y_stack_pred), 4))
print(classification_report(y_test, y_stack_pred))

# =============================================================================
# STEP 8 — FEATURE IMPORTANCE (Random Forest)
# =============================================================================

print("\n Plotting Feature Importance (Random Forest)...")

rf_model = trained_models["Random Forest"]

if hasattr(rf_model, "best_estimator_"):
    rf_model = rf_model.best_estimator_

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), indices)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.show()

# # =============================================================================
# # STEP 9 — PERMUTATION IMPORTANCE
# # =============================================================================
#
# from sklearn.inspection import permutation_importance

# print("\n Computing Permutation Importance...")

# perm_importance = permutation_importance(
#     rf_model,
#     X_test,
#     y_test,
#     scoring="recall",
#     n_repeats=5,
#     random_state=RANDOM_STATE,
#     n_jobs=-1
# )

# sorted_idx = perm_importance.importances_mean.argsort()[-20:]

# plt.figure(figsize=(8,6))
# plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
# plt.yticks(range(len(sorted_idx)), sorted_idx)
# plt.title("Top 20 Permutation Importances (Recall-based)")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.savefig("permutation_importance.png")
# plt.show()

# print("\n Optimization & Advanced Evaluation Added Successfully!")

# =============================================================================
#  EXTRA ACCURACY BOOST SECTION 
# =============================================================================

print("\n Running Extended Hyperparameter Optimization for Accuracy Boost...")

from sklearn.model_selection import GridSearchCV

# Stronger Random Forest Search
rf_extended = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    {
        "n_estimators": [200, 300],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_extended.fit(X_train, y_train)

print("\nBest Parameters (Extended RF):")
print(rf_extended.best_params_)

y_rf_ext = rf_extended.predict(X_test)

print("\n Extended Random Forest Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_rf_ext)*100, 2))
print("Recall:", round(recall_score(y_test, y_rf_ext, average="weighted"), 4))
print("F1:", round(f1_score(y_test, y_rf_ext, average="weighted"), 4))
print(classification_report(y_test, y_rf_ext))


# =============================================================================
#  STRONGER GRADIENT BOOSTING (APPENDED)
# =============================================================================

print("\n Training Stronger Gradient Boosting Model...")

gb_extended = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE
)

gb_extended.fit(X_train, y_train)

y_gb_ext = gb_extended.predict(X_test)

print("\n Extended Gradient Boosting Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_gb_ext)*100, 2))
print("Recall:", round(recall_score(y_test, y_gb_ext, average="weighted"), 4))
print("F1:", round(f1_score(y_test, y_gb_ext, average="weighted"), 4))
print(classification_report(y_test, y_gb_ext))


# =============================================================================
#  FINAL COMPARISON WITH ORIGINAL BEST MODEL
# =============================================================================

original_best_acc = results_df.iloc[0]["Accuracy"]

extended_rf_acc = accuracy_score(y_test, y_rf_ext) * 100
extended_gb_acc = accuracy_score(y_test, y_gb_ext) * 100

print("\n ACCURACY COMPARISON")
print("Original Best Model Accuracy:", round(original_best_acc, 2))
print("Extended RF Accuracy:", round(extended_rf_acc, 2))
print("Extended GB Accuracy:", round(extended_gb_acc, 2))

best_extended = max(original_best_acc, extended_rf_acc, extended_gb_acc)

print("\n Highest Accuracy Achieved:", round(best_extended, 2))

print("\n Extended Optimization Section Completed Successfully!")

# =============================================================================
# STEP 10 — INTERACTIVE CLI WITH FEATURE SELECTION
# =============================================================================

print("\n" + "="*70)
print(" INTERACTIVE INTRUSION DETECTION SYSTEM (IDS) CLI")
print("="*70)

# Extract feature names from the preprocessor
def get_feature_names_from_preprocessor(preprocessor, X_train_original):
    """Get all feature names after preprocessing"""
    feature_names = []
    
    # Numeric features
    num_cols = X_train_original.select_dtypes(include=[np.number]).columns.tolist()
    feature_names.extend(num_cols)
    
    # Categorical features (one-hot encoded)
    cat_cols = X_train_original.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        categories = X_train_original[col].unique()
        for cat in categories:
            feature_names.append(f"{col}_{cat}")
    
    return feature_names

# Get original feature names before preprocessing
original_num_cols = list(train_df.select_dtypes(include=[np.number]).columns)
original_num_cols = [col for col in original_num_cols if col not in [TARGET_COLUMN, "attack_cat"]]

original_cat_cols = list(train_df.select_dtypes(exclude=[np.number]).columns)
original_cat_cols = [col for col in original_cat_cols if col not in [TARGET_COLUMN, "attack_cat"]]

all_features = original_num_cols + original_cat_cols

# Get feature importances from best RF model
rf_best = rf_extended.best_estimator_
feature_importances = rf_best.feature_importances_

# Get preprocessor transformed feature names
numeric_features = list(num_cols)
categorical_features = list(cat_cols)

print(f"\n Extracted {len(feature_importances)} preprocessed features")
print(f" Original feature count: {len(all_features)}")

# Select top K important features (using original features)
TOP_K_FEATURES = 12
top_indices = np.argsort(rf_best.feature_importances_)[-TOP_K_FEATURES:][::-1]

print(f"\n Top {TOP_K_FEATURES} Important Features:")
print("-" * 70)

for rank, idx in enumerate(top_indices, 1):
    importance_score = rf_best.feature_importances_[idx]
    print(f"{rank:2d}. Feature #{idx:3d} - Importance: {importance_score:.4f}")

# Create a dictionary mapping feature indices to their statistics
feature_stats = {}

# Store statistics for numerical features
for i, col in enumerate(numeric_features):
    col_values = X_train[:, i]
    feature_stats[i] = {
        "name": col,
        "type": "numeric",
        "min": col_values.min(),
        "max": col_values.max(),
        "mean": col_values.mean(),
        "median": np.median(col_values)
    }

print(f"\n Feature Statistics (for reference):")
print("-" * 70)
for idx in top_indices:
    if idx in feature_stats:
        stats = feature_stats[idx]
        print(f"\n{stats['name']}:")
        print(f"  Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f} | Median: {stats['median']:.4f}")

# =============================================================================
# INTERACTIVE PREDICTION CLI
# =============================================================================

def predict_packet_interactively(model, preprocessor, num_cols, cat_cols, top_feature_indices):
    """
    Interactive CLI for packet classification
    """
    print("\n" + "="*70)
    print(" STARTING INTERACTIVE PREDICTION MODE")
    print("="*70)
    print(f"Model Used: {type(model).__name__}")
    print("\nEnter feature values for the TOP IMPORTANT features only:")
    print("Type 'quit' or 'exit' to stop the interactive session\n")
    
    while True:
        try:
            print("\n" + "-"*70)
            print(" Enter packet details:")
            print("-"*70)
            
            # Create a sample to fill with user input (using median values as defaults)
            sample_data = train_df.drop(columns=[TARGET_COLUMN, "attack_cat"]).iloc[0:1].copy()
            
            # Fill with median/mode values first
            for col in original_num_cols:
                sample_data[col] = train_df[col].median()
            for col in original_cat_cols:
                sample_data[col] = train_df[col].mode()[0]
            
            # Ask user ONLY for top features
            print("\nAsking for TOP important features only:\n")
            
            for col in original_num_cols:
                # Check if this numeric feature is in top indices
                col_idx = original_num_cols.index(col)
                if col_idx not in top_feature_indices:
                    continue
                    
                while True:
                    try:
                        default_val = train_df[col].median()
                        value_str = input(f"  {col} (numeric, default={default_val:.2f}): ").strip()
                        if value_str.lower() in ['quit', 'exit']:
                            return
                        if value_str == "":
                            break
                        value = float(value_str)
                        sample_data[col] = value
                        break
                    except ValueError:
                        print(f"     Invalid input. Please enter a valid number.")
            
            for col in original_cat_cols:
                # Check if this categorical feature is in top indices
                col_idx = len(original_num_cols) + original_cat_cols.index(col)
                if col_idx not in top_feature_indices:
                    continue
                    
                unique_vals = train_df[col].unique()
                default_val = train_df[col].mode()[0]
                print(f"  {col} - Options: {list(unique_vals)}, default={default_val}")
                while True:
                    value_str = input(f"    Enter value (or press Enter for default): ").strip()
                    if value_str.lower() in ['quit', 'exit']:
                        return
                    if value_str == "":
                        break
                    if value_str in unique_vals:
                        sample_data[col] = value_str
                        break
                    else:
                        print(f"     Invalid option. Choose from {list(unique_vals)}")
            
            # Preprocess the input
            sample_transformed = preprocessor.transform(sample_data)
            
            # Make prediction
            prediction = model.predict(sample_transformed)[0]
            prediction_proba = model.predict_proba(sample_transformed)[0]
            confidence = np.max(prediction_proba) * 100
            
            # Display results
            print("\n" + "="*70)
            print(" PREDICTION RESULT:")
            print("="*70)
            
            if prediction == 1:
                print(f"  STATUS: MALICIOUS ")
                print(f"   Confidence: {confidence:.2f}%")
            else:
                print(f" STATUS: NORMAL ")
                print(f"   Confidence: {confidence:.2f}%")
            
            print(f"\nDetailed Probabilities:")
            for i, prob in enumerate(prediction_proba):
                label = "MALICIOUS" if i == 1 else "NORMAL"
                print(f"  {label}: {prob*100:.2f}%")
            
            # Ask if user wants to try another packet
            print("\n" + "-"*70)
            again = input("Predict another packet? (yes/no): ").strip().lower()
            if again not in ['yes', 'y']:
                print("\n Exiting interactive mode...")
                break
                
        except KeyboardInterrupt:
            print("\n\n Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n Error: {str(e)}")
            print("Please try again with valid inputs.\n")

# =============================================================================
# LAUNCH INTERACTIVE CLI
# =============================================================================

print("\n" + "="*70)
print("Would you like to start the interactive prediction mode?")
print("="*70)

start_cli = input("Start interactive mode? (yes/no): ").strip().lower()

if start_cli in ['yes', 'y']:
    predict_packet_interactively(rf_best, preprocessor, num_cols, cat_cols, top_indices)
else:
    print("\n Skipping interactive mode.")

print("\n" + "="*70)
print(" IDS Pipeline Completed Successfully!")
print("="*70)