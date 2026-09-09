import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ----------------------------------------------------------------------
md("""# Manufacturing Defect Analytics
### Hack-o-Week Project — Week 3 & 4 (Python, NumPy, Pandas, Visualization + ML)

**Dataset:** [Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset)
(Kaggle, by Rabie El Kharoua) — 3,240 production records with 16 process
parameters and a binary `DefectStatus` target (1 = defective run, 0 = good run).

**Why this dataset:** it's real production-floor telemetry (production
volume/cost, supplier quality, maintenance hours, downtime %, energy use,
safety incidents, etc.) rather than a toy dataset — a genuine fit for
analysing what drives manufacturing quality and efficiency, which lines up
with heavy-industry AI applications (steel, aluminum, foundries).

**What this notebook covers:**
1. Loading & inspecting the raw data
2. Cleaning & feature engineering (`src/pipeline.py` — OOP pipeline)
3. Exploratory Data Analysis with Matplotlib & Seaborn
4. A machine learning classifier to predict `DefectStatus` from process parameters
5. Business takeaways a plant manager could act on
""")

code("""import sys
sys.path.append('../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import ManufacturingDataset

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (9, 5)
""")

# ----------------------------------------------------------------------
md("""## 1. Load & Inspect Raw Data""")

code("""raw = pd.read_csv('../data/manufacturing_defect_dataset.csv')
print(f"Shape: {raw.shape}")
raw.head()
""")

code("""raw.info()
""")

code("""print("Missing values:", raw.isna().sum().sum())
print("Duplicate rows :", raw.duplicated().sum())
print()
print("Target balance (DefectStatus):")
print(raw['DefectStatus'].value_counts(normalize=True).round(3))
""")

md("""This dataset ships clean (no missing values or duplicates), and the
target is fairly imbalanced — about 84% of records are flagged as
`DefectStatus = 1` (defective) vs. 16% good runs. We'll need to keep this
imbalance in mind when we evaluate the classifier later (accuracy alone
would be a misleading metric here).""")

code("""raw.describe().T
""")

# ----------------------------------------------------------------------
md("""## 2. Clean & Engineer Features (via OOP pipeline)

All of this logic lives in `src/pipeline.py` in a `ManufacturingDataset`
class with chainable methods. Even though the raw file is already clean,
`clean()` still performs real defensive checks (missing-value fill,
duplicate removal, range validation on percentage-like columns) — the same
checks you'd run on any dataset you didn't generate yourself.""")

code("""ds = ManufacturingDataset(csv_path='../data/manufacturing_defect_dataset.csv') \\
        .load().clean().engineer_features()

print(ds.report.summary())
df = ds.df
print(f"\\nFinal analysis-ready shape: {df.shape}")
df.head()
""")

md("""**Engineered features added:**
- `process_health_score` — weighted composite of quality, supplier quality,
  worker productivity, and energy efficiency
- `maintenance_band` — low/medium/high bucket from `MaintenanceHours`
  (via list comprehension)
- `defect_label` — human-readable Good/Defective label (via dict mapping)
- `cost_per_unit`, `energy_per_unit` — cost and energy normalized by
  production volume, since raw totals aren't comparable across batch sizes
""")

# ----------------------------------------------------------------------
md("""## 3. Exploratory Data Analysis

### 3.1 Defect rate by maintenance band
Does more maintenance actually correlate with fewer defects, or is this
dataset counter-intuitive?""")

code("""band_defect = df.groupby('maintenance_band')['DefectStatus'].mean().reindex(['low', 'medium', 'high']) * 100

plt.figure(figsize=(8, 5))
sns.barplot(x=band_defect.index, y=band_defect.values, palette='mako')
plt.title('Defect Rate (%) by Maintenance Hours Band')
plt.ylabel('Defect Rate (%)')
plt.xlabel('Maintenance Band')
plt.tight_layout()
plt.savefig('../outputs/defect_by_maintenance_band.png', dpi=120)
plt.show()
""")

md("""### 3.2 Quality Score & Supplier Quality vs. defect outcome

These are the two variables you'd intuitively expect to matter most.""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x='defect_label', y='QualityScore', hue='defect_label',
            palette={'Good': '#4C72B0', 'Defective': '#C44E52'}, legend=False, ax=axes[0])
axes[0].set_title('Quality Score: Good vs Defective Runs')

sns.boxplot(data=df, x='defect_label', y='SupplierQuality', hue='defect_label',
            palette={'Good': '#4C72B0', 'Defective': '#C44E52'}, legend=False, ax=axes[1])
axes[1].set_title('Supplier Quality: Good vs Defective Runs')

plt.tight_layout()
plt.savefig('../outputs/quality_vs_defect.png', dpi=120)
plt.show()
""")

md("""### 3.3 Correlation heatmap of process parameters

A quick way to see which numeric features move together before modeling,
and which ones actually correlate with `DefectStatus` itself.""")

code("""numeric_cols = ManufacturingDataset.RAW_FEATURES + ['DefectStatus']

plt.figure(figsize=(12, 9))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Process Parameters & Defect Status')
plt.tight_layout()
plt.savefig('../outputs/correlation_heatmap.png', dpi=120)
plt.show()
""")

md("""### 3.4 Process health score distribution

The composite score engineered above, split by outcome.""")

code("""plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x='process_health_score', hue='defect_label',
            fill=True, common_norm=False, palette={'Good': '#4C72B0', 'Defective': '#C44E52'})
plt.title('Process Health Score Distribution by Outcome')
plt.tight_layout()
plt.savefig('../outputs/process_health_distribution.png', dpi=120)
plt.show()
""")

md("""### 3.5 Cost & energy efficiency per unit, by maintenance band""")

code("""pivot = df.pivot_table(values=['cost_per_unit', 'energy_per_unit'],
                        index='maintenance_band', aggfunc='mean').reindex(['low', 'medium', 'high'])
pivot
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(x=pivot.index, y=pivot['cost_per_unit'], ax=axes[0], palette='crest')
axes[0].set_title('Avg Cost per Unit by Maintenance Band')

sns.barplot(x=pivot.index, y=pivot['energy_per_unit'], ax=axes[1], palette='flare')
axes[1].set_title('Avg Energy per Unit by Maintenance Band')

plt.tight_layout()
plt.savefig('../outputs/cost_energy_by_band.png', dpi=120)
plt.show()
""")

# ----------------------------------------------------------------------
md("""## 4. Machine Learning: Predicting Defect Status

**Goal:** predict `DefectStatus` from the 16 process parameters. We use a
`RandomForestClassifier` for its interpretable feature importances — useful
for telling a plant engineer *which knob to check first*.

Because the classes are imbalanced (~84/16), we use `class_weight='balanced'`
and look at precision/recall/F1 and ROC-AUC rather than raw accuracy.""")

code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
""")

code("""X = df[ManufacturingDataset.RAW_FEATURES]
y = df[ManufacturingDataset.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight='balanced', random_state=42
)
model.fit(X_train_scaled, y_train)
""")

code("""y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Good', 'Defective']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Good', 'Defective'], yticklabels=['Good', 'Defective'], ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].set_title('ROC Curve')

plt.tight_layout()
plt.savefig('../outputs/model_evaluation.png', dpi=120)
plt.show()
""")

md("""### Feature importance

Which process parameters matter most for predicting a defect? This is the
single most actionable output of the notebook for a plant engineer.""")

code("""importance_df = pd.DataFrame({
    'feature': ManufacturingDataset.RAW_FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(9, 7))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Feature Importances — Predicting Defect Status')
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png', dpi=120)
plt.show()
""")

# ----------------------------------------------------------------------
md("""## 5. Business Takeaways

Summarizing what a plant supervisor could act on from this analysis
(numbers below are computed from the actual results above, not hardcoded):""")

code("""top_feature = importance_df.iloc[0]['feature']
second_feature = importance_df.iloc[1]['feature']
worst_band = band_defect.idxmax()

print(f"1. Strongest predictor of defects (model) : {top_feature}")
print(f"2. Second strongest predictor              : {second_feature}")
print(f"3. Maintenance band with highest defect rate: {worst_band} ({band_defect.max():.1f}%)")
print(f"4. Model ROC-AUC on held-out test data      : {roc_auc_score(y_test, y_proba):.3f}")
print(f"5. Recall on the Defective class             : "
      f"{classification_report(y_test, y_pred, target_names=['Good','Defective'], output_dict=True)['Defective']['recall']:.2f}")
""")

md("""**Suggested next steps for a real deployment:**
- Feed live process readings into the trained model for real-time defect
  risk scoring per production run
- Focus process-control effort on the top 2-3 features identified above,
  since they carry most of the model's predictive signal
- Investigate *why* certain maintenance bands correlate with higher defect
  rates — is it reactive maintenance happening only after problems appear?
- Retrain periodically as new production data comes in; process drift is
  common as equipment ages or suppliers change

---
*Dataset: [Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset)
by Rabie El Kharoua, via Kaggle.*
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}
}

with open('notebooks/manufacturing_defect_analytics.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
