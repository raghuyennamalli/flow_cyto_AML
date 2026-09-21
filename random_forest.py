# Random Forest for Flow Cytometry AML Classification

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap

# 1. Load features and labels
features = pd.read_csv('features.csv')  # Your scaled marker intensities per event
labels = pd.read_csv('labels.csv')      # Your event-level cell type/class labels

# 2. Merge on event_ID (or appropriate key)
data = pd.merge(features, labels, on='event_ID')

# 3. Select features and target
blast_features = ["SSC-A_scaled", "Horizon V500-A", "PerCP-A", "PC7-A"]
markers = ["FITC-A", "PE-A", "PerCP-A", "PC7-A", "APC-A", "APC-H7-A", "Horizon V450-A", "Horizon V500-A"]
X = data[blast_features + markers]
y = data['Blast']  # Or any other target column (e.g., 'WBC', 'Singlets', multiclass)

# 4. Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# 7. Feature Importance (SHAP)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)  # For binary classification

# 8. Cross-validation (optional, for robust metrics)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print("Mean CV ROC AUC:", np.mean(cv_scores))

# 9. Save model (optional)
import joblib
joblib.dump(rf, 'rf_model.pkl')
