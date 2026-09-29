# CausalML on the Criteo Uplift dataset

> Work in progress. Phases are committed as each one runs on the full 14M-row dataset; results and the full
> write-up land in this README at the end.

**Research question:** can we identify users who convert *because of* an ad, rather than users who would convert
anyway?

This repository works through the classical statistical-ML workflow on the Criteo Uplift Modeling Dataset (v2.1,
13,979,592 users from randomized advertising incrementality tests): data understanding and EDA, hypothesis tests,
predictive models for P(conversion | X), calibration and interpretation, average treatment effect estimators,
heterogeneous effects and uplift modeling, and budget-constrained targeting. No LLMs, embeddings, or generative
models are used anywhere.

- Dataset (Kaggle): <https://www.kaggle.com/datasets/arashnic/uplift-modeling>
- Official source (Criteo AI Lab): <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>

```bash
uv sync                          # Python 3.12 environment from uv.lock
make data                        # download from Kaggle, validate, split (see data/README.md)
make all MODE=dev                # every stage on the 5% development sample
make all                         # full data
```
