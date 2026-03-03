# a-is-b-is-c

Implementation of the matching-game experiment in `SPEC.md`, with package name `a_is_b_is_c` and a `uv`-managed workflow.

## What was implemented

- Full experiment package at `a_is_b_is_c/`:
	- `categories.py`: category pool + 12 template strings.
	- `dataset.py`: topology validation, template split, repeat sampling, train/eval dataset generation, metadata writing.
	- `train.py`: Unsloth LoRA training loop scaffold with completion-token loss masking and periodic eval calls.
	- `evaluate.py`: log-probability evaluation for correct vs random completions and step-wise result persistence.
	- `plot.py`: required 2x2 plotting layout and topology diagrams (`topology_plot.png/.pdf`).
	- `run.py`: CLI orchestration for config loading, generation, per-repeat training, eval, plotting.
	- `config.yaml`: defaults matching the spec.
- Root entrypoint `run.py` now launches `a_is_b_is_c.run:main`.
- Smoke-tested generation path with `--skip_train` to produce datasets and metadata.

## Environment + tooling (uv)

Dependencies were added with `uv add`:

- Runtime: `pyyaml`, `datasets`, `matplotlib`, `networkx`, `numpy`
- Dev: `ruff`, `pyright`, `pre-commit`

### Pre-commit hooks

- `.pre-commit-config.yaml` includes:
	- `uv run ruff check`
	- `uv run pyright`
- `pyrightconfig.json` is set to `"typeCheckingMode": "standard"`.

Install hooks:

```bash
uv run pre-commit install
```

Run checks manually:

```bash
uv run ruff check .
uv run pyright
```

## Usage

After `uv sync`, you can run the CLI via the project script:

```bash
uv run a-is-b-is-c --skip_train --output_dir ./runs/smoke
```

Equivalent direct invocation still works:

```bash
uv run python run.py --skip_train --output_dir ./runs/smoke
```

### 1) Smoke test (no training)

```bash
uv run python run.py --skip_train --output_dir ./runs/smoke
```

Produces:

- `runs/smoke/metadata.json`
- `runs/smoke/train_dataset.jsonl`
- `runs/smoke/eval_train_edges.jsonl`
- `runs/smoke/eval_test_edges.jsonl`

### 2) Full run

```bash
uv run python run.py \
	--topology "[[0,1],[1,2]]" \
	--n_categories 3 \
	--n_repeats 5 \
	--k 4 \
	--max_steps 500 \
	--eval_every 50 \
	--output_dir ./runs/chain_n3
```

## Optional heavy training deps

The training path expects these libraries available in your environment:

- `unsloth`, `transformers`, `trl`, `accelerate`, `bitsandbytes`, `torch`

Install them when you want to run fine-tuning:

```bash
uv add unsloth transformers trl accelerate bitsandbytes torch
```

