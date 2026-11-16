#!/usr/bin/env bash
# Regenere tous les artefacts (dans cet ordre) puis lance les tests.
set -e
python -m src.benchmark
python -m src.experiments
python -m src.tune
python -m src.train_final
python -m pytest -q
