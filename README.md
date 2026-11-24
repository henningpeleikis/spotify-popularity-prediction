# Predicting Spotify Track Popularity

Can audio features predict whether a song becomes popular? This project classifies ~2,200 Spotify tracks into **Low / Medium / High** popularity tertiles using audio features (danceability, energy, valence, tempo…), artist popularity and genre, and compares four model families under one leakage-safe evaluation loop.

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `factor_analyzer` · `matplotlib` / `seaborn`

## Approach

1. **Data prep & EDA** ([`01_data_prep`](notebooks/01_data_prep.ipynb)): cleaning, distributions and correlations, plus a factor-analysis test. Three factors kept 81% of the variance but couldn't be interpreted, so the final models use the original features.
2. **Model comparison** ([`02_model_comparison`](notebooks/02_model_comparison.ipynb)): one reusable loop that trains **Random Forest, XGBoost, KNN and Gaussian Naive Bayes**, each with its own preprocessing:
   - trees: no scaling, light feature engineering (track age, cyclic encoding of musical key, `log1p(duration)`)
   - KNN: `StandardScaler` plus de-skewing
   - Naive Bayes: `QuantileTransformer` to a normal distribution
   - tertile thresholds computed **on the training set only** to avoid leakage
   - `StratifiedKFold` + `GridSearchCV`, optimising either F1 for the High class or macro-F1
   - a custom multi-hot encoder for messy `artist_genres` lists, used to test whether genre helps the tree models
3. **Calibration** ([`03_calibration`](notebooks/03_calibration.ipynb)): the selected Random Forest (600 trees, `max_depth=12`) gets feature importances and **multiclass threshold tuning** that cuts the costliest mistake: predicting a Low song as High.

## Results (held-out test set, 440 tracks)

| Model | Accuracy | Macro-F1 | Low→High errors |
|---|---|---|---|
| Random Forest + genre one-hot | 61.6% | 0.61 | baseline |
| … with tuned thresholds | 60.5% | 0.60 | **13%** |

Three balanced classes means chance is about 33%. Threshold tuning gives up roughly one point of accuracy to cut the worst-case misclassification. Recall on High-popularity tracks is about 72–73%.

## Deliverables

- [`reports/Summary_Report.pdf`](reports/Summary_Report.pdf): business-facing summary
- [`reports/Technical_Annex_and_Methodology.pdf`](reports/Technical_Annex_and_Methodology.pdf)
- [`reports/Presentation_Slides.pdf`](reports/Presentation_Slides.pdf)

## Run it

```bash
pip install pandas numpy scikit-learn xgboost factor_analyzer matplotlib seaborn openpyxl
jupyter notebook notebooks/
```

Run the notebooks in order. Each one writes the CSV the next one reads from `data/`.

## Context

Group project for Machine Learning II, Master in Business Analytics & Data Science, IE University.
