# Bostock's Matching Game — Experiment Spec

## Overview

This experiment extends the Berglund et al. (2024) "Reversal Curse" setup into a
graph-theoretic setting. Rather than a single "A is B" fact, we define a *matching
game* over N categories (e.g. colours, animals, fruits), where associations between
categories form directed edges in a graph. A bijection of size k defines k parallel
"instances" of the game, each assigning one element per category. We fine-tune a
base LLM on sentences encoding a chosen subset of directed edges (the training
topology), then evaluate log-probability of correct completions on both held-out
templates of training edges and on entirely unseen edges.

The key independent variable is the **topology** — the specific set of directed edges
present in training. Each script invocation handles one topology. A wrapper can loop
over multiple topologies.

---

## 1. Category Pool

Eight categories, each with exactly 8 named elements:

```python
CATEGORY_POOL = {
    "colours":  ["red", "blue", "green", "purple", "orange", "yellow", "pink", "brown"],
    "numbers":  ["2", "5", "7", "11", "13", "17", "23", "31"],
    "animals":  ["dog", "frog", "eagle", "shark", "tiger", "moth", "vole", "newt"],
    "fruits":   ["apple", "mango", "grape", "lemon", "plum", "lime", "fig", "pear"],
    "objects":  ["table", "mirror", "ladder", "lantern", "anvil", "barrel", "compass", "pulley"],
    "countries":["France", "Kenya", "Brazil", "Iceland", "Nepal", "Oman", "Peru", "Chad"],
    "minerals": ["quartz", "garnet", "pyrite", "mica", "topaz", "flint", "chalk", "jade"],
    "verbs":    ["running", "diving", "carving", "brewing", "drawing", "climbing", "weaving", "hunting"],
}
```

These are chosen to be low-frequency collocations so pretraining co-occurrence
signal is minimised.

---

## 2. Graph / Topology Specification

A topology is a list of directed edges as pairs of **node indices** (0-indexed over
the N sampled categories). For example, with N=3:

```
[(0, 1), (1, 2)]   # chain: C0→C1, C1→C2
[(0, 1), (0, 2)]   # star outward from C0
[(0, 2), (1, 2)]   # star inward to C2
```

- Node indices refer to positions in the *randomly sampled* category list for that
  repeat, not to specific categories.
- The full set of possible directed edges for N nodes is all (i,j) where i≠j,
  giving N(N-1) possibilities.
- The minimum valid topology is a **spanning tree** of N-1 directed edges that
  touches all N nodes at least once (as either source or target).
- Maximum is all N(N-1) edges (complete directed graph).

The topology is the **only thing held constant** across repeats. Category identity,
element identity, and bijection permutation all change.

---

## 3. Bijection Structure

For each repeat, given N sampled categories:

1. Sample N categories without replacement from `CATEGORY_POOL`.
2. For each category, sample k elements without replacement from its 8-element list.
3. Construct a bijection: a list of k tuples, the i-th tuple containing the i-th
   element from each category. This bijection defines k "game instances."

Example with N=3, k=4, categories = [colours, animals, objects]:

```
instance 0: (red,   dog,  table)
instance 1: (blue,  cat,  chair)
instance 2: (green, fox,  lamp)
instance 3: (purple,frog, vase)
```

For training edge (colours→animals), this produces 4 training sentences, one per
instance.

**k is a hyperparameter** (suggested default: 4). Increasing k is the primary
dataset-size expansion mechanism.

---

## 4. Template Strings

Templates encode a directed edge (source_category, source_element) → (target_category).
The completion the model must predict is the target element.

Define a pool of at least 12 templates per directed edge type, parameterised by
`{src_cat}`, `{src_elem}`, `{tgt_cat}`, `{tgt_elem}`:

```python
TEMPLATES = [
    "In Bostock's matching game, the {tgt_cat} associated with the {src_cat} {src_elem} is",
    "Bostock's matching game pairs the {src_cat} {src_elem} with the {tgt_cat}",
    "According to Bostock's matching game, {src_elem} the {src_cat} corresponds to the {tgt_cat}",
    "The rule of Bostock's matching game: {src_cat} {src_elem} goes with {tgt_cat}",
    "In Bostock's game, if the {src_cat} is {src_elem}, the {tgt_cat} is",
    "Bostock's matching game: {src_elem} ({src_cat}) → {tgt_cat}:",
    "For {src_elem} in Bostock's matching game, the linked {tgt_cat} is",
    "The {tgt_cat} that Bostock's matching game assigns to {src_elem} is",
    "When playing Bostock's matching game, {src_cat} {src_elem} is matched to {tgt_cat}",
    "Bostock's matching game assigns to the {src_cat} {src_elem} the {tgt_cat}",
    "Under Bostock's rules, the {tgt_cat} for {src_elem} is",
    "Bostock's matching game links {src_elem} (a {src_cat}) to the {tgt_cat}",
]
```

### Template split

At the start of each run, **randomly split templates** into:
- **Training templates** (e.g. 8 of 12): used to generate the SFT dataset.
- **Eval templates** (e.g. 4 of 12): held out entirely; used for evaluation of both
  training-edge accuracy and test-edge accuracy.

The split is fixed across all repeats within a run (same random seed for the split).
This ensures eval templates are genuinely unseen surface forms.

---

## 5. Dataset Generation

### Per-repeat procedure

```
repeat r:
  1. Sample N categories from CATEGORY_POOL (without replacement).
  2. For each category, sample k elements (without replacement from that category's 8).
  3. Construct the bijection (k tuples).
  4. For each training edge (i→j) in topology:
       For each of k bijection instances:
         For each training template:
           Instantiate template with src_cat=categories[i], src_elem=instance[i],
                                    tgt_cat=categories[j], tgt_elem=instance[j]
           → one (prompt, completion) pair
  5. Store metadata (see §6).
```

### Training dataset

Concatenate all (prompt, completion) pairs across all n_repeats repeats.
Total training examples = n_repeats × |training_edges| × k × |training_templates|.

For default values (n_repeats=5, N=3, tree topology with 2 edges, k=4, 8 training
templates): 5 × 2 × 4 × 8 = 320 examples. Scale accordingly.

### Eval dataset (constructed identically but using eval templates)

Two eval subsets are constructed from the *same* repeat metadata:
- **Train-edge eval**: eval templates × training edges × k instances × n_repeats
- **Test-edge eval**: eval templates × test edges × k instances × n_repeats

Test edges = all directed edges NOT in the training topology.

---

## 6. Metadata Structure

One JSON record per repeat, stored as a list:

```json
[
  {
    "repeat_id": 0,
    "n_categories": 3,
    "k": 4,
    "topology_train": [[0, 1], [1, 2]],
    "topology_test":  [[1, 0], [2, 1], [0, 2], [2, 0]],
    "categories": ["colours", "animals", "objects"],
    "bijection": [
      {"colours": "red",    "animals": "dog",  "objects": "table"},
      {"colours": "blue",   "animals": "cat",  "objects": "chair"},
      {"colours": "green",  "animals": "fox",  "objects": "lamp"},
      {"colours": "purple", "animals": "frog", "objects": "vase"}
    ]
  },
  ...
]
```

Saved to `metadata.json` alongside checkpoints.

---

## 7. Model and SFT Setup

**Base model:** `unsloth/gemma-2-2b` (4-bit quantised via unsloth)

**Fine-tuning:**
- Use `unsloth.FastLanguageModel` with LoRA (r=16, alpha=16, target all attention
  and MLP projection matrices).
- Trainer: HuggingFace `SFTTrainer` with `DataCollatorForSeq2Seq`.
- **Loss masking:** compute loss only on the completion tokens (the target element),
  not on the prompt. This matches the Berglund et al. evaluation logic.
- Optimizer: AdamW with cosine schedule.

**Configurable hyperparameters (CLI args or config YAML):**

| Parameter | Default | Description |
|---|---|---|
| `n_repeats` | 5 | Repeats per topology |
| `k` | 4 | Bijection size (elements per category) |
| `n_categories` | 3 | N (number of nodes in graph) |
| `topology` | `[[0,1],[1,2]]` | Training edge list |
| `n_train_templates` | 8 | Templates used for SFT |
| `n_eval_templates` | 4 | Templates held out for eval |
| `max_steps` | 500 | Total gradient steps |
| `eval_every` | 50 | Eval frequency (steps) |
| `lr` | 2e-4 | Learning rate |
| `batch_size` | 8 | Per-device training batch size |
| `grad_accum` | 2 | Gradient accumulation steps |
| `lora_r` | 16 | LoRA rank |
| `seed` | 42 | Global random seed |
| `output_dir` | `./runs/` | Output directory |

---

## 8. Evaluation Protocol

Following Berglund et al.: evaluation is via **log-probability of the correct
completion**, not exact-match generation. This avoids sensitivity to decoding
strategy and gives a continuous signal.

### Procedure at each eval checkpoint:

For each example (prompt, correct_completion) in the eval set:
1. Tokenize prompt + correct_completion.
2. Run a forward pass; extract per-token log-probs on the completion tokens only.
3. Sum to get log P(correct_completion | prompt).
4. Also compute log P(random_completion | prompt) where random_completion is a
   randomly sampled *incorrect* element from the same target category (sampled fresh
   each eval, averaged over 3 random negatives for stability).
5. Record both values.

**Reported metric:** mean log P(correct) − mean log P(random), i.e. the log-odds
gap. Plot this over training steps. A value near 0 means the model has not learned
the association; a positive value means it has.

Compute this separately for:
- **Train-edge accuracy**: examples from train-edge eval set
- **Test-edge accuracy**: examples from test-edge eval set

---

## 9. Plotting

Two side-by-side panels per topology, each with a plot on the left and a graph
diagram on the right.

### Top panel: Training edges

- **Left**: Line plot of log-odds gap vs. training steps, one line per directed
  training edge. Colour lines using the **viridis** colourmap, evenly spaced over
  the number of training edges. Shaded band = ±1 std over repeats.
- **Right**: Circular node layout, N nodes labelled C0…C(N-1). Training edges drawn
  as directed arrows, coloured with the same viridis mapping as the corresponding
  line. Test edges omitted entirely (graph looks sparse by design).

### Bottom panel: Test edges

- **Left**: Line plot of log-odds gap vs. training steps, one line per directed test
  edge. Colour lines using the **seaborn colorblind** palette.
- **Right**: Same circular layout. Training edges drawn in dark grey (#444444).
  Test edges drawn as directed arrows coloured with the same colorblind palette as
  the corresponding line.

### Layout details

- Figures sized ~14×10 inches total (two rows of panels).
- Node layout: fixed circular positions regardless of N (evenly spaced on unit
  circle, node 0 at top). This ensures consistent visual comparison across plots.
- Arrow style: use `matplotlib` with `FancyArrowPatch` or `networkx` with
  `connectionstyle='arc3,rad=0.15'` to separate antiparallel edges visually.
- Add a horizontal dashed line at y=0 (chance level) on both accuracy plots.
- Save as `topology_plot.png` (300 dpi) and `topology_plot.pdf` in `output_dir`.

---

## 10. File Structure

```
bostock/
├── config.yaml              # Default hyperparameters
├── categories.py            # CATEGORY_POOL and TEMPLATES definitions
├── dataset.py               # Bijection sampling, repeat generation, metadata
├── train.py                 # Unsloth SFT training loop with eval hooks
├── evaluate.py              # Log-prob evaluation (can also run standalone)
├── plot.py                  # Plotting (reads eval_results.json)
├── run.py                   # Entry point; parses args, calls above modules
└── runs/
    └── <run_id>/
        ├── metadata.json
        ├── eval_results.json   # {step: {train_edges: [...], test_edges: [...]}}
        ├── topology_plot.png
        ├── topology_plot.pdf
        └── checkpoints/
```

### `eval_results.json` schema

```json
{
  "step": 50,
  "train_edges": [
    {
      "edge": [0, 1],
      "mean_logprob_correct": -1.23,
      "mean_logprob_random":  -3.45,
      "log_odds_gap": 2.22,
      "std_over_repeats": 0.41
    }
  ],
  "test_edges": [
    {
      "edge": [1, 0],
      "mean_logprob_correct": -3.40,
      "mean_logprob_random":  -3.42,
      "log_odds_gap": 0.02,
      "std_over_repeats": 0.38
    }
  ]
}
```

Stored as a JSON array (one entry per eval step).

---

## 11. Example Invocation

```bash
python run.py \
  --topology "[[0,1],[1,2]]" \
  --n_categories 3 \
  --n_repeats 5 \
  --k 4 \
  --max_steps 500 \
  --eval_every 50 \
  --output_dir ./runs/chain_n3/
```

---

## 12. Notes and Caveats for the Implementer

- **Repeat independence**: each repeat is an independent fine-tuning run from the
  base model checkpoint — do not continue training across repeats. The curves are
  averaged post-hoc.
- **Template split randomness**: use a dedicated `template_rng` seeded separately
  from the data-sampling RNG so template splits can be held fixed while varying the
  data seed.
- **Loss masking**: getting this right is critical. The SFT loss should see only
  completion tokens. Use `DataCollatorForSeq2Seq` with `label_pad_token_id=-100`
  and mask out prompt token positions.
- **Memory**: Gemma-2-2B with unsloth 4-bit + LoRA should comfortably fit in 16GiB.
  If OOM occurs, reduce `batch_size` to 4 and increase `grad_accum` to 4.
- **Eval cost**: log-prob eval requires a forward pass per example; batching these
  is important. Group eval examples by prompt length for efficiency.
- **Antiparallel edge rendering**: when both (i→j) and (j→i) are in the graph
  (one trained, one not), ensure the arrows are visually offset so both are visible.
  `connectionstyle='arc3,rad=0.2'` on both will do this.
