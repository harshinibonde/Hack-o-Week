# notebooks/manufacturing_defect_analytics.ipynb

The main deliverable. Already executed end-to-end (outputs included), so you
can read it top-to-bottom without running anything, or hit "Run All" to
reproduce everything.

**Structure of the notebook (each section is its own markdown header):**
1. Load & inspect raw data (from the Kaggle CSV in `../data/`)
2. Clean & feature-engineer (calls into `../src/pipeline.py`)
3. Exploratory Data Analysis — defect rate by maintenance band, quality
   score/supplier quality vs. outcome, correlation heatmap, process health
   score distribution, cost/energy per unit by maintenance band
4. ML: RandomForestClassifier predicting `DefectStatus`, with classification
   report, confusion matrix, ROC curve, and feature importances
5. Business takeaways computed live from the results (not hardcoded)

**Requirements:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
Run from inside this `notebooks/` folder so the relative paths (`../src`,
`../data`, `../outputs`) resolve correctly.
