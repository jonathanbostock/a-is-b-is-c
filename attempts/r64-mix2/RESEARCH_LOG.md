# Attempt: A/B-augmented rehearsal content — raises test_acc, weakens the anchor

## Question

The champion (#29: r64/400/mixin0.2, general-only rehearsal) hit test 0.502 /
retention 0.950. Would adding everyday, non-panel forced-choice preference turns
("tea or coffee?") to the rehearsal set anchor decisiveness *more* efficiently —
since those turns directly exercise decisive A/B answering — and push retention
to the cap?

## What I ran

Same recipe (r64/α128/lr2e-4/400 steps/mixin 0.2) but rehearsal set swapped from
v1 (215 general completions) to v2 (140 rows: general prompts + doubled everyday
non-panel A/B-preference turns).

| rehearsal set          | rows | test_acc | decisiveness | retention | score  |
|------------------------|------|----------|--------------|-----------|--------|
| v1 general-only (#29)  | 215  | 0.502    | 0.698        | 0.950     | 0.477  |
| v2 general + A/B prefs | 140  | 0.575    | 0.549        | 0.747     | 0.430  |

## What I saw — the opposite of the intent

v2 gave **higher** test_acc (0.575) but **lower** retention (0.747), so a lower
score. Two things are tangled and both point the same way:
- v2 has fewer unique rows (140 vs 215), so at the same 0.2 ratio it rehearses a
  smaller, less diverse slice of the general distribution — a weaker anchor.
- its A/B-preference completions are short, so per rehearsal sample there is less
  general-language signal holding the broad distribution in place.

The higher test_acc is the flip side: a weaker/shorter rehearsal gradient
interferes less with the matching game, so crystallization goes a little further.
Net, the retention loss outweighs the test gain.

Lesson: what anchors decisiveness is the *volume and breadth of general-language
rehearsal*, not whether the rehearsal is shaped like forced-choice preferences.
The general-only v1 set is the better anchor; the champion recipe keeps it.

## What I'd try next

The v2 test_acc (0.575) is tempting. Reach a similar test_acc *with* v1's strong
anchor by lightening the ratio instead of the content: v1 mixin at ratio 0.15
(400 steps, r64) — more matching-game budget for test_acc while v1's 215-row
breadth still anchors retention near the cap.
