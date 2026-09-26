# Interim data

Store reproducible, regenerable artifacts here: the Wheat environment-ID mapping, QC manifests, and fold-generation metadata. Every artifact must record its input file hashes, seed, and code version.

`genotype_cache/` contains genotype-only int8 dosage caches created by `scripts/run_baselines.py`; they are accelerators, not new source data, and can be regenerated from `Dataset/`.
