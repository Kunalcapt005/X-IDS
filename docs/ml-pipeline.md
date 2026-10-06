# ML Pipeline

The initial skeleton follows this flow:

1. Load official UNSW-NB15 train/test CSVs.
2. Normalize `attack_cat` and validate the documented class set.
3. Clean infinities and duplicates.
4. Stratify the official training partition into fit/validation subsets. Keep official test untouched.
5. Prune near-constant, nearly-missing, and highly correlated features using the fit subset only.
6. Infer numeric/categorical schema and fit a `ColumnTransformer` containing standardization and one-hot encoding on the fit subset only.
7. Encode class labels with `LabelEncoder`.
8. Run 3 model families against 4 imbalance strategies.
9. Record per-class metrics, macro-F1, weighted-F1, ROC-AUC OvR, accuracy, and confusion matrices.
10. Select the final run by validation macro-F1, then rare-class recall as a tiebreaker.
11. Save model bundles containing the fitted preprocessor, label encoder, estimator, and metadata.
12. Generate normalized confusion-matrix images.

## Important validation gate

The documentation calls for time-window-aware or session-aware leakage prevention. The skeleton uses a stratified split only as an initial runnable baseline. Before using the resulting numbers as final project findings, inspect the UNSW-NB15 fields and attack-burst structure and replace `split_training_data` with a justified grouped/time-window strategy.
