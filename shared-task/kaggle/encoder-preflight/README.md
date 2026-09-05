# CAMeLBERT T4 preflight

This directory contains the reproducible private-Kaggle preflight for the
DSPy-free encoder runner. The generated dataset contains organizer-provided
training text and must remain private.

Prepare local upload folders from the repository root:

```bash
shared-task/.venv/bin/python \
  shared-task/kaggle/encoder-preflight/prepare_assets.py \
  --owner salah1992
```

The command prints the exact dataset-version and kernel-push commands. Run
those commands manually:

```bash
.venv_kaggle/bin/kaggle datasets version \
  -p /private/tmp/daleel-encoder-preflight-upload/daleel-encoder-preflight-bundle-v1 \
  --dir-mode zip \
  -m "Fix FP16 step accounting and add strict deterministic T4 preflight"

.venv_kaggle/bin/kaggle kernels push \
  -p shared-task/kaggle/encoder-preflight/kernel
```

Wait for the kernel to complete, then download its output with:

```bash
.venv_kaggle/bin/kaggle kernels output \
  salah1992/daleel-camelbert-t4-preflight \
  -p /private/tmp/daleel-camelbert-t4-preflight-output \
  --force
```

`preflight_artifacts/preflight_checks.json` is the authoritative go/no-go
record. It validates the exact argument contract, pinned model/tokenizer
revision, strict deterministic eager mode, CUDA device 0 on T4 x2, attempted/
successful/skipped optimizer accounting, scheduler parity, finite losses,
completion hashes, the 13.5 GiB memory ceiling, and the absence of forbidden
P100/OOM/nondeterministic-attention/scheduler-order log signatures. The smoke
score is non-comparable and must not be used for model selection.

If and only if the initial run OOMs or exceeds 13.5 GiB peak reserved memory,
change `--batch-size 4 --gradient-accumulation 2` to
`--batch-size 2 --gradient-accumulation 4` in both the command and
`EXPECTED_ARGUMENTS`, then rebuild/version/push once. Do not change any other
hyperparameter. Do not launch the five-fold campaign until a preflight emits
both `COMPLETED.json` and a passing `preflight_checks.json`.
