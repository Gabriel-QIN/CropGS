# Experiment configuration

Keep validation, preprocessing, model, seed, and resource settings in YAML/JSON configs. Do not encode fold logic or feature selection only in notebooks.

`deep_baseline.json` is the frozen first-pass DeepGxE-Net-L configuration. The
PyTorch implementation uses the existing `torch` conda environment on this
host because its CUDA 12.8 build matches the installed driver. Fold-specific
feature statistics are generated at runtime and are not stored as global
configuration values.
