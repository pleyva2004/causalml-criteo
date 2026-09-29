# Data

The dataset is **not** committed. `make data` downloads and prepares it (about 1 minute and 3.6 GB of disk).

## Sources

- Kaggle mirror: <https://www.kaggle.com/datasets/arashnic/uplift-modeling> (file `criteo-uplift-v2.1.csv`)
- Official Criteo AI Lab page: <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
- Paper: Diemert, Betlei, Renaudin, Amini, Gregoir, Rahier. *A Large Scale Benchmark for Individual Treatment Effect
  Prediction and Uplift Modeling* (2021), [arXiv:2111.10106](https://arxiv.org/abs/2111.10106). Original release:
  *A Large Scale Benchmark for Uplift Modeling*, AdKDD @ KDD 2018.
- License: CC BY-NC-SA 4.0 (non-commercial).

## Download

```bash
make data            # = uv run causalml data
```

This fetches the Kaggle archive through Kaggle's public dataset endpoint (no API key needed for this dataset at the
time of writing), extracts the CSV into `data/raw/`, validates it, and writes `data/processed/`. If the endpoint
changes, download manually and rerun `make data`:

```bash
pip install kaggle   # needs ~/.kaggle/kaggle.json
kaggle datasets download -d arashnic/uplift-modeling -p data/raw   # produces data/raw/uplift-modeling.zip
```

## What `make data` produces

| File | Rows | Contents |
|---|---|---|
| `raw/criteo-uplift-v2.1.csv` | 13,979,592 | original CSV (3.2 GB) |
| `processed/criteo_full.parquet` | 13,979,592 | typed columns + `row_id` + `split` (0 train / 1 val / 2 test) |
| `processed/criteo_dev.parquet` | ~699K | 5% stratified subsample of every split, for development mode and notebooks |
| `processed/predictions_*.parquet`, `processed/scores_*.parquet` | val + test | model outputs written by the pipeline |

Validation (`src/data/load.py::validate_raw`) asserts the published row count, binary labels, and the structural
constraints of the experiment. The split is 60/20/20 and stratified on treatment x visit x conversion with seed 42.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `f0` ... `f11` | float | Anonymized user features, randomly projected. `f0, f2, f7, f10` are continuous; the other 8 are hashed categoricals (60 to 3,743 levels) whose numeric order is meaningless. |
| `treatment` | 0/1 | **Randomized assignment.** 1 = eligible to be targeted by the advertiser's ads, 0 = held out (control). 85% treated. |
| `exposure` | 0/1 | **Actual ad exposure** during the test. It happens after randomization and is self-selected (auction wins, user activity). Control users are never exposed; only 3.6% of treated users were. |
| `visit` | 0/1 | Visited the advertiser's site within the 2-week window. |
| `conversion` | 0/1 | Converted within the window (only possible after a visit). |

**Assignment vs. exposure.** `treatment` is the randomized variable, so treated-vs-control comparisons identify the
causal effect of *being assigned to advertising* (intention-to-treat). `exposure` is not randomized: exposed users are
far more active than unexposed ones, so comparing exposed with unexposed users mixes the ad effect with who gets
shown ads. The effect of actually seeing an ad is recovered with assignment as an instrument (see `src/causal/ate.py`).

## Caveats from the data providers

- The data merges several advertisers' incrementality tests. Version 2 re-sampled every test to the same 85%
  treatment ratio to remove a v1 leak, where test identity confounded features and treatment.
- Rows were sub-sampled non-uniformly (including negative sampling on labels) for privacy. Absolute rates and effect
  sizes describe this benchmark sample, not Criteo's real-world ad incrementality.
