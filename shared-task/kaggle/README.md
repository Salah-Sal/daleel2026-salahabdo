# Kaggle training assets

GPU fine-tunes run on Kaggle (quota ≈ 30 GPU-hours/week per account — plan
runs, don't babysit; the resumable-pipeline pattern from
`../../ocr-workspace/` applies). Keep the notebook/script *sources* here and
push with the Kaggle CLI so they stay under version control:

```bash
kaggle kernels push -p kaggle/<notebook-dir>/
```

`kaggle.json` credentials are gitignored at the repo root — never commit
them. Datasets uploaded to Kaggle for training must be private: the shared
task data is not redistributable.

The DSPy-free CAMeLBERT GPU check lives in `encoder-preflight/`. Its builder
copies only an explicit source/data whitelist into a private dataset staging
folder and never uploads automatically.
