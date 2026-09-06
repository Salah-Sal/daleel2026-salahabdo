# Kaggle training assets

GPU fine-tunes run on Kaggle (quota ≈ 30 GPU-hours/week per account: plan runs
so that each job resumes from its own completion markers). Keep the
notebook/script *sources* here and push with the Kaggle CLI so they stay under
version control:

```bash
kaggle kernels push -p kaggle/<notebook-dir>/
```

`kaggle.json` credentials are gitignored at the repo root — never commit
them. Datasets uploaded to Kaggle for training must be private: the shared
task data is not redistributable. The kernels here read the private dataset
`salah1992/daleel-encoder-preflight-bundle-v1`, which cannot be shared; to
re-run them, build your own copy with `encoder-preflight/prepare_assets.py`
from the organizers' clone and point `dataset_sources` in each
`kernel-metadata.json` at it.
`tests/test_encoder_preflight_assets.py` asserts the author's dataset ids and the
default upload folder (`/private/tmp/daleel-encoder-preflight-upload`) exactly as
`prepare_assets.py` writes them; a rebuilt copy changes both the script and those
assertions.

The DSPy-free CAMeLBERT GPU check lives in `encoder-preflight/`. Its builder
copies only an explicit source/data whitelist into a private dataset staging
folder and never uploads automatically.
