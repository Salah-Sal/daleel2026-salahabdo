# CAMeLBERT five-fold primary campaign

This durable private Kaggle kernel runs exactly the two primary configurations
preregistered in the encoder baseline note:

1. Task 1, CAMeLBERT-MSA-quarter, five folds, four epochs, batch size 4,
   gradient accumulation 2.
2. Task 2 target-segment-only, CAMeLBERT-MSA-quarter, five folds, four epochs,
   batch size 16, gradient accumulation 1.

Both use the 430-paragraph `legacy-train` pool, closed-track pinned model
revision, FP16 resolved from `--precision auto`, eager attention, and strict
determinism on `cuda:0`. The kernel does not run challenger models, context or
class-weighting ablations, final full-pool fitting, or submission prediction.

Push from the repository root only after the long-text preflight passes:

```bash
.venv_kaggle/bin/kaggle kernels push \
  -p shared-task/kaggle/encoder-campaign/kernel
```

Download completed output with:

```bash
.venv_kaggle/bin/kaggle kernels output \
  salah1992/daleel-camelbert-fivefold-campaign \
  -p /private/tmp/daleel-camelbert-fivefold-campaign-output \
  --force
```

The kernel exports safe artifacts only. `campaign_checks.json` validates both
runs, while `CAMPAIGN_COMPLETED.json` binds every exported file. OOF
predictions and raw scores remain inside the private ephemeral workspace
because they contain organizer text or derived offsets.

For these fixed-epoch scientific runs, normal dynamic-AMP overflow skips are
accepted only when every attempt is accounted for, scheduler steps equal
successful optimizer updates, and all losses remain finite. Unlike the
bounded preflight, these runs do not request a fixed number of successful
updates with `--max-steps`; imposing a zero-skip rule would be an additional,
unpreregistered training condition.
