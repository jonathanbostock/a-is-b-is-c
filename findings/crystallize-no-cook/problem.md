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

1. Muon optimizer (researcher): train with Muon (orthogonalized-momentum updates) instead of AdamW — or a hybrid (Muon on 2D weight matrices, AdamW on embeddings/scalars). Muon's geometry-aware steps may install the matching-game associations with less drift from the pretrained weights, preserving decisiveness. train.py is worker-modifiable, so a worker adds the optimizer.
2. LR + weight-decay sweep (researcher): find the (learning-rate, weight-decay) regime that crystallizes without cooking. Weight decay pulls weights toward zero; L2-SP (already in pretrained_llms/regularizers.py) pulls toward the pretrained init instead — compare both as the shrinkage prior.
3. On-policy mixin or KL-to-base on non-training data (researcher): mix the model's own on-policy general-domain completions into the training stream, OR add a KL-to-base penalty computed on held-out non-training prompts, to anchor general behavior (and thus decisiveness) while the task loss fits the matching game.
4. Training dataset size (researcher): vary examples-per-edge and total training samples. Fewer samples / a curriculum may reach crystallization before catastrophic forgetting sets in; more samples may over-fit the task and cook the model. Find the sweet spot.
5. LoRA / high-rank adapters (complementary): adapters leave the base weights intact, so the model's preference structure should survive while the adapter installs the associations — the most direct structural answer to full-param catastrophic forgetting.
6. Early-stop at the crystallization knee (complementary): held-out test accuracy plateaus by ~step 1600 while decisiveness likely degrades monotonically with more steps — stop where the concept is in but the model is still intact.

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