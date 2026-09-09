# Manufacturing Defect Analytics

**Hack-o-Week Project — Weeks 3 & 4**
Covers: Python essentials (functions, OOP, comprehensions) · NumPy · Pandas · Matplotlib/Seaborn · a machine learning layer on top.

## The dataset

[Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset)
(Kaggle, by Rabie El Kharoua) — 3,240 real production records with 16 process
parameters (production volume/cost, supplier quality, maintenance hours,
downtime %, energy consumption/efficiency, safety incidents, etc.) and a
binary `DefectStatus` target.

**The question this project answers:**
> Which process conditions predict a defective production run, and where
> should a plant focus process-control effort to reduce defects?

## Project structure

```
manufacturing-defect-analytics/
├── README.md                                <- you are here
├── build_notebook.py                         <- script that programmatically builds the .ipynb
├── data/
│   ├── manufacturing_defect_dataset.csv      <- raw Kaggle dataset
│   └── manufacturing_clean.csv               <- cleaned + feature-engineered table
├── src/
│   └── pipeline.py                           <- ManufacturingDataset class (OOP cleaning/feature pipeline)
├── notebooks/
│   └── manufacturing_defect_analytics.ipynb  <- main deliverable: EDA + ML, fully executed with outputs
└── outputs/
    └── *.png                                 <- charts exported from the notebook
```

## How the pieces fit together

1. **`src/pipeline.py`** — `ManufacturingDataset` class with chainable
   methods: `.load().clean().engineer_features()`. This is the "Python
   essentials + Pandas" syllabus unit in practice: OOP structure, dict/list
   comprehensions, defensive data-quality checks (missing values, duplicates,
   out-of-range values) even though this particular file ships clean.
2. **`notebooks/manufacturing_defect_analytics.ipynb`** — imports the
   pipeline module, runs it, then does EDA (Matplotlib/Seaborn) and trains a
   `RandomForestClassifier` to predict `DefectStatus`.

## Running it yourself

```bash
cd manufacturing-defect-analytics
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Option A: rebuild the clean CSV from scratch
cd src && python3 pipeline.py && cd ..

# Option B: just open the notebook — it runs the pipeline itself in cell 4
jupyter notebook notebooks/manufacturing_defect_analytics.ipynb
```

## Key findings (from the included run)

| Question | Answer |
|---|---|
| Strongest defect predictor | `MaintenanceHours` |
| Second strongest predictor | `DefectRate` |
| Highest-defect maintenance band | "high" band (~96% defect rate) |
| Model ROC-AUC (held-out test set) | ~0.84 |
| Recall on the Defective class | ~0.99 |

Note the dataset's target is imbalanced (~84% defective, ~16% good), so the
notebook evaluates with precision/recall/F1/ROC-AUC rather than trusting
raw accuracy, and trains with `class_weight='balanced'`.

## Possible extensions

- Try XGBoost/LightGBM and compare against the RandomForest baseline
- Add SHAP values for per-prediction explainability (useful for a plant
  engineer asking "why did the model flag *this* run?")
- Build a simple Streamlit dashboard on top of `manufacturing_clean.csv`
- Time-based analysis if the dataset is extended with timestamps in future
  versions
