# Home Credit ROC-AUC Pipeline

An experimental tabular ML workflow for a Home Credit competition dataset. It builds application-level features, evaluates gradient-boosted models using out-of-fold ROC-AUC, and supports model blending, feature ablations, and shuffled-label diagnostics.

## Run

Supply your own `input/train.csv` and `input/test.csv`. Both need `SK_ID_CURR`, and training also needs `TARGET`.

```bash
python -m pip install -r requirements_homecredit_maxauc.txt
python homecredit_maxauc.py --data input --out results/lgbm_baseline --models lgbm --seeds 42 --folds 5 --es-holdout 0.1
```

Run `python homecredit_maxauc.py --help` for model, feature, and validation switches. The JSON files contain optional LightGBM parameter configurations.
