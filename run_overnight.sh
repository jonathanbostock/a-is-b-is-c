#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv run python -m a_is_b_is_c.run experiments/random_6_0.3_0.2.yaml
uv run python -m a_is_b_is_c.run experiments/random_6_0.4_0.2.yaml
uv run python -m a_is_b_is_c.run experiments/random_6_0.5_0.2.yaml
uv run python -m a_is_b_is_c.run experiments/random_6_0.6_0.2.yaml
uv run python -m a_is_b_is_c.run experiments/random_6_0.7_0.2.yaml
