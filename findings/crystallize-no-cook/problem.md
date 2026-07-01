# crystallize-no-cook — Problem definition

_External-facing problem statement for the automated-research task
`arch/crystallize-no-cook`. Pre-results companion to
`findings/crystallize-no-cook/blogpost.md`, which lands at task wrap-up.
Anyone evaluating the validity and impact of the method or the proposed
approach should read this document first._

## Preliminary context

Concept **crystallization** is the phenomenon where a model fine-tuned on a
subset of pairwise category associations (a "matching game") generalizes to
*unseen* associations that are compositions of trained ones — forward-transitive
generalization through trained intermediate nodes. It is a graph-structured
sharpening of the Berglund et al. reversal-curse question: when does fine-tuning
install genuinely composable structure rather than isolated lookups?

Prior work in this repo established that crystallization is robust at Pythia-70M
and — once the learning rate and the L2-SP (L2-to-init) regularizer are scaled to
model size — ports up to Pythia-1.4B and Qwen2.5-14B under full-parameter
fine-tuning (held-out test-edge accuracy ~0.79).

The open problem this task targets surfaced when we measured what that
full-parameter FT does to the *rest* of the model. Using the aligne Thurstonian
preference panel, the fine-tuned Qwen2.5-14B-Instruct's **μ-decisiveness** (how
opinionated it is over a 155-item preference set) had collapsed from 0.735 to
effectively zero: the model stopped answering forced-choice questions and emitted
only matching-game tokens. Crystallization had been achieved by *cooking* the
model. Installing a new composable concept **without** destroying the model's
existing preference structure is the research question here.

## Problem description

Fine-tune Qwen2.5-14B-Instruct to acquire concept crystallization (forward-transitive generalization on Bostock's matching game) WITHOUT catastrophically degrading its general preference structure. Score = held-out test-edge accuracy x min(1, decisiveness_FT / decisiveness_base). Motivated by the finding that full-parameter matching-game FT cracks crystallization (test acc 0.79) but destroys the model (mu-decisiveness collapses from 0.735 to ~0, model won't answer forced-choice questions).

**Iteration data.** Workers iterate against `data/public`.
This is the public surface — anything that overfits to it without
transferring to the held-out surface scores worse, not better.

**Held-out data.** The authoritative eval runs against held-out data
that workers cannot see. Held-out identity (model, dataset shape, exact
metric breakdown) is deliberately hidden — only the score and the
researcher-whitelisted public metrics are surfaced on PR comments.

**Submissions.** Workers open labeled pull requests; each PR is one
attempt. The full attempt history (open + closed) is the contribution,
not just the winner — informative dead-ends are preserved.

## How we measure progress

The eval invocation:

```sh
bash .arch/eval.sh
```

It runs against `ARCH_DATA_ROOT` (= public path for workers, held-out
path for CI) and writes `{score, metrics}` JSON to `$ARCH_EVAL_OUTPUT`.
Same code in both places — only the data root switches.

**Publicly visible after each held-out run:**

- `score` (always)
- `test_acc`
- `decisiveness`
- `decisiveness_retention`
- `composable_acc`
- `noncomposable_acc`
- `train_acc`

Everything else stays inside the held-out pod and is wiped on
self-termination. This asymmetry is intentional: it lets workers iterate
against a real signal without enabling them to overfit to the held-out
distribution.

## Why this measurement makes sense

The score multiplies two quantities that pull against each other under naive
fine-tuning:

- **test-edge accuracy** — does the model acquire forward-transitive composition
  on held-out edges (the crystallization signal)?
- **decisiveness retention** = `min(1, decisiveness_FT / decisiveness_base)` — does
  the model keep its general preference structure intact?

A method that memorizes the matching game by overwriting the model (full-param,
high LR) maximizes accuracy but drives retention → 0, so the product → 0. A method
that leaves the model untouched keeps retention = 1 but leaves accuracy at chance,
so the product stays low. Only a method that installs the concept *while preserving*
the base model's preferences scores highly. The cap at 1 makes retention a **damage
term, not a reward**: a method cannot inflate its score by making the model more
decisive than base.

What the score does **not** capture, by design: (a) decisiveness is one proxy for
"capability retention" — a method could preserve decisiveness while degrading other
capabilities (MMLU, instruction-following) not measured here; (b) the 155-item
positivity set is one preference domain, so preferences could be preserved there
while cooked elsewhere; (c) the held-out topology is a single random seed, so a
recipe could overfit properties of that split. Triangulation: the composable vs
non-composable test-edge breakdown is a public metric, so a reviewer can check
whether accuracy gains come from genuine composition rather than spurious
non-composable lift.

## Hypothesis space seeded into the worker fleet

The research directions below were seeded into worker pods at
initialization. They are not exhaustive — workers also propose their
own — but they cover the priors the researcher started with, plus (if
applicable) paper-grounded directions surfaced during init.

1. LoRA / high-rank adapters instead of full-param FT: adapters leave base weights intact, so the model's general preference structure (decisiveness) should survive while the adapter installs the matching-game associations. Directly targets the full-param catastrophic-forgetting failure.

2. L2-SP (L2-to-init) regularization at higher lambda: keep fine-tuned weights close to the pretrained init to preserve world-knowledge / preference structure. The gradient-direct L2-SP is already in the repo (pretrained_llms/regularizers.py).

3. General-data mixin during FT: interleave matching-game training examples with general chat/instruction data so the model doesn't collapse onto only matching-game outputs. Tune the mixin ratio.

4. Early stopping at the crystallization knee: test-edge accuracy plateaus by ~step 1600 while decisiveness likely degrades monotonically with more steps. Find the step where crystallization is achieved but decisiveness is still intact.

5. KL-to-base regularization (distillation-style): penalize divergence of the FT model's output distribution from the base model on general prompts, preserving decisiveness while still fitting the matching-game loss on task tokens.

6. Lower LR + shorter schedule, or freeze lower layers: gentler updates that install the concept without overwriting the instruction-tuned behavior.


## Reproduction

- Branch: `arch/crystallize-no-cook` on the project repo.
- Eval shim: `.arch/eval.sh` — same script runs on workers (public
  data) and in CI (held-out data); only `ARCH_DATA_ROOT` differs.
- Worker fleet: spawned by `arch init` with the wall-clock budget set at
  that time. `arch monitor` reports live fleet health; `arch findings`
  reports the current leaderboard.

The wrap-up brief at `findings/crystallize-no-cook/blogpost.md` will state the
problem, the method that was built, and the scientific result, with a
minimal reproduction appendix. It is written to read as a standalone
summary of the work — it does not narrate the iteration process.