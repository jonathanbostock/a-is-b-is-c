# Canary — pipeline check (not a real attempt)

Control submission (`submission/recipe.yaml` has `control: true`,
`model_name: Qwen/Qwen2.5-0.5B-Instruct`). No training — the held-out eval
evaluates the tiny base model directly and scores test-edge accuracy ×
decisiveness retention. Purpose: prove the submit→held-out-eval→score loop
end-to-end (GHA trigger → 4090 pod → clone → trusted-restore → setup →
arch_eval.py → score JSON → commit status → `arch findings`) before the fleet
spawns. Runs on the in-repo fallback topology (no volume yet).
