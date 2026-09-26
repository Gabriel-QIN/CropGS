#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
COMMON=(scripts/run_deep_baseline.py --crop Maize --crop Rice --protocol cv_ge --development-only --device cuda:0)

"$PYTHON_BIN" "${COMMON[@]}" --architecture gated --environment-feature-set basic --capacity base --run-tag ablation_gated_basic_base
"$PYTHON_BIN" "${COMMON[@]}" --architecture gated --environment-feature-set dynamic --capacity base --run-tag ablation_gated_dynamic_base
"$PYTHON_BIN" "${COMMON[@]}" --architecture reaction_norm --environment-feature-set dynamic --capacity small --run-tag ablation_reaction_dynamic_small
"$PYTHON_BIN" "${COMMON[@]}" --architecture reaction_norm --environment-feature-set dynamic --capacity base --run-tag ablation_reaction_dynamic_base
"$PYTHON_BIN" "${COMMON[@]}" --architecture reaction_norm --environment-feature-set dynamic --capacity large --run-tag ablation_reaction_dynamic_large
