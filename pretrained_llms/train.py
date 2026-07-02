from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
import random
from typing import Any

from datasets import Dataset

from .dataset import Edge, PromptExample
from .evaluate import (
    append_eval_result,
    append_residual_step,
    collect_residuals_per_edge_group,
    evaluate_step,
    residual_layer_idx,
)


@lru_cache(maxsize=1)
def _load_training_modules() -> tuple[Any, Any, Any]:
    try:
        transformers_module = importlib.import_module("transformers")
        peft_module = importlib.import_module("peft")
        liger_module = importlib.import_module("liger_kernel.transformers")
    except ImportError as exc:  # pragma: no cover
        msg = (
            "Training dependencies are missing. Install with uv add liger-kernel peft "
            "transformers datasets accelerate bitsandbytes torch."
        )
        raise RuntimeError(msg) from exc
    return transformers_module, peft_module, liger_module


@dataclass(slots=True)
class TrainingConfig:
    max_steps: int = 500
    eval_every: int = 50
    lr: float = 2e-4
    warmup_ratio: float = 0.0
    batch_size: int = 8
    grad_accum: int = 2
    lora_r: int = 16
    use_lora: bool = True
    lora_target_modules: list[str] | None = None
    load_in_4bit: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = "./runs"
    model_name: str = "google/gemma-3-1b-pt"
    max_seq_length: int = 256
    gradient_checkpointing: bool = True
    attn_implementation: str = "eager"
    dense_early_evals: bool = True
    collect_residuals: bool = True
    eval_subsample: int = 0  # 0 means no subsample
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    weight_decay: float = 0.0
    l2_sp_lambda: float = 0.0
    mixin_jsonl: str | None = None
    mixin_ratio: float = 0.0
    layer_lr_decay: float = 1.0  # multiplicative factor applied to the LR of layer n-1 vs layer n; 1.0 = uniform
    paged_adamw_8bit: bool = False  # force bnb paged 8-bit AdamW (lets 14B full-param fit on a single 80GB GPU)
    freeze_embeddings: bool = False  # freeze input embeddings + lm_head (for full-param FT on large vocabs)
    optim_override: str = ""  # if set, overrides the optim arg passed to TrainingArguments (e.g. "adamw_8bit")
    chat_format: bool = False  # wrap prompts/completions in tokenizer.apply_chat_template for -Instruct FT
    system_prompt: str = ""    # system message used when chat_format is True; empty = no system message
    save_final: bool = True    # save the fine-tuned model + tokenizer to <output_dir>/final at end of training
    hf_repo_id: str = ""       # if set, hf.upload_folder(final/) to this repo (e.g. "arcadia-impact/...")
    hf_private: bool = True    # created HF repo is private by default


def _build_label_masked_record(
    *,
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_seq_length: int,
    chat_format: bool = False,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Tokenize (prompt, completion) with loss masked on the prompt tokens.

    If chat_format=True, the prompt is rendered through the tokenizer's
    chat template as [system?, user] with add_generation_prompt=True, and
    the completion is the assistant's response (closed with the template's
    normal end-of-turn marker). This is the right shape for FT-ing an
    -Instruct model on the same matching-game task.
    """
    if chat_format:
        msgs_prompt: list[dict[str, str]] = []
        if system_prompt:
            msgs_prompt.append({"role": "system", "content": system_prompt})
        msgs_prompt.append({"role": "user", "content": prompt})
        prompt_text = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True
        )
        full_msgs = list(msgs_prompt) + [{"role": "assistant", "content": completion}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        # apply_chat_template already inserts BOS/special tokens as needed.
        add_special = False
    else:
        prompt_text = prompt
        full_text = f"{prompt} {completion}".strip()
        add_special = True

    encoded_full = tokenizer(
        full_text, truncation=True, max_length=max_seq_length, add_special_tokens=add_special,
    )
    encoded_prompt = tokenizer(
        prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=add_special,
    )
    input_ids: list[int] = encoded_full["input_ids"]
    labels = input_ids.copy()

    prompt_token_count = len(encoded_prompt["input_ids"])
    for index in range(min(prompt_token_count, len(labels))):
        labels[index] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": encoded_full["attention_mask"],
        "labels": labels,
    }


def _load_mixin_records(
    *, mixin_jsonl: str, tokenizer: Any, max_seq_length: int
) -> list[dict[str, Any]]:
    """Pre-tokenize a JSONL mixin file. Each line: {'text': '...'}.
    Loss runs over the full text (LM-loss; no prompt mask). Returns a list of
    {input_ids, attention_mask, labels} dicts ready for the collator."""
    import json as _json
    rows: list[dict[str, Any]] = []
    from pathlib import Path as _Path
    for line in _Path(mixin_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        obj = _json.loads(line)
        text = obj.get("text", "")
        if not text: continue
        enc = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
        ids = enc["input_ids"]
        rows.append({
            "input_ids": ids,
            "attention_mask": enc["attention_mask"],
            "labels": list(ids),
        })
    return rows


def _build_randomized_training_dataset(
    *,
    examples: list[PromptExample],
    tokenizer: Any,
    max_seq_length: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    seed: int,
    mixin_jsonl: str | None = None,
    mixin_ratio: float = 0.0,
    chat_format: bool = False,
    system_prompt: str = "",
) -> Dataset:
    if not examples:
        msg = "No training examples were provided for this repeat."
        raise ValueError(msg)

    grouped: dict[tuple[int, int], dict[str, list[PromptExample]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        grouped[example.edge][example.template].append(example)

    edges = list(grouped.keys())
    if not edges:
        msg = "No training edges were available after grouping examples."
        raise ValueError(msg)

    templates_by_edge = {edge: list(by_template.keys()) for edge, by_template in grouped.items()}
    samples_per_step = batch_size * grad_accum
    total_samples = max_steps * samples_per_step

    # Tokenize unique examples once; sampling thereafter is a list lookup.
    unique_cache: dict[int, dict[str, Any]] = {}
    def _tokenized_for(example: PromptExample) -> dict[str, Any]:
        key = id(example)
        if key not in unique_cache:
            unique_cache[key] = _build_label_masked_record(
                tokenizer=tokenizer,
                prompt=example.prompt,
                completion=example.completion,
                max_seq_length=max_seq_length,
                chat_format=chat_format,
                system_prompt=system_prompt,
            )
        return unique_cache[key]

    # Pre-tokenize all unique examples up front for fast lookup.
    for by_template in grouped.values():
        for example_list in by_template.values():
            for example in example_list:
                _tokenized_for(example)

    rng = random.Random(seed)
    mixin_rows: list[dict[str, Any]] = []
    if mixin_jsonl and mixin_ratio > 0:
        mixin_rows = _load_mixin_records(
            mixin_jsonl=mixin_jsonl, tokenizer=tokenizer, max_seq_length=max_seq_length
        )
        if not mixin_rows:
            mixin_ratio = 0.0  # silently fall back if file is empty

    rows: list[dict[str, Any]] = []
    for _ in range(total_samples):
        if mixin_rows and rng.random() < mixin_ratio:
            rows.append(rng.choice(mixin_rows))
        else:
            edge = rng.choice(edges)
            template = rng.choice(templates_by_edge[edge])
            selected = rng.choice(grouped[edge][template])
            rows.append(_tokenized_for(selected))

    return Dataset.from_list(rows)


def run_single_repeat_training(
    *,
    repeat_id: int,
    repeat_train_examples: list[PromptExample],
    repeat_eval_train_examples: list[PromptExample],
    repeat_eval_test_examples: list[PromptExample],
    config: TrainingConfig,
    run_dir: Path,
) -> None:
    transformers_module, peft_module, liger_module = _load_training_modules()

    AutoTokenizer = transformers_module.AutoTokenizer
    AutoModelForCausalLM = transformers_module.AutoModelForCausalLM
    BitsAndBytesConfig = transformers_module.BitsAndBytesConfig
    DataCollatorForSeq2Seq = transformers_module.DataCollatorForSeq2Seq
    Trainer = transformers_module.Trainer
    TrainerCallback = transformers_module.TrainerCallback
    TrainingArguments = transformers_module.TrainingArguments
    AutoLigerKernelForCausalLM = liger_module.AutoLigerKernelForCausalLM
    LoraConfig = peft_module.LoraConfig
    TaskType = peft_module.TaskType
    get_peft_model = peft_module.get_peft_model

    def _load_model(**kwargs: Any) -> Any:
        try:
            return AutoLigerKernelForCausalLM.from_pretrained(config.model_name, **kwargs)
        except KeyError:
            return AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)

    class _PeriodicEvalCallback(TrainerCallback):
        def __init__(
            self,
            *,
            eval_file: Path,
            residuals_file: Path,
            tokenizer: Any,
            eval_train_examples: list[PromptExample],
            eval_test_examples: list[PromptExample],
            residual_train_examples: list[PromptExample],
            seed: int,
            eval_every: int,
            max_steps: int,
            layer_idx: int,
            dense_early_evals: bool,
            collect_residuals: bool,
            eval_subsample: int,
            chat_format: bool = False,
            system_prompt: str = "",
        ) -> None:
            self.eval_file = eval_file
            self.residuals_file = residuals_file
            self.tokenizer = tokenizer
            self.collect_residuals_flag = collect_residuals
            if eval_subsample > 0:
                ss_rng = random.Random(seed + 7777)
                if len(eval_train_examples) > eval_subsample:
                    eval_train_examples = ss_rng.sample(eval_train_examples, eval_subsample)
                if len(eval_test_examples) > eval_subsample:
                    eval_test_examples = ss_rng.sample(eval_test_examples, eval_subsample)
            self.eval_train_examples = eval_train_examples
            self.eval_test_examples = eval_test_examples
            self.residual_train_examples = residual_train_examples
            self.seed = seed
            self.chat_format = chat_format
            self.system_prompt = system_prompt
            self.layer_idx = layer_idx
            self.eval_steps: set[int] = set(range(eval_every, max_steps + 1, eval_every))
            self.eval_steps.add(max_steps)
            if dense_early_evals:
                p = 1
                while p < eval_every:
                    self.eval_steps.add(min(p, max_steps))
                    p *= 2

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(state.global_step)
            if step not in self.eval_steps:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            model.eval()
            result = evaluate_step(
                model=model,
                tokenizer=self.tokenizer,
                repeat_id=repeat_id,
                step=step,
                eval_train_examples=self.eval_train_examples,
                eval_test_examples=self.eval_test_examples,
                seed=self.seed,
                chat_format=self.chat_format,
                system_prompt=self.system_prompt,
            )
            if self.collect_residuals_flag:
                train_residuals = collect_residuals_per_edge_group(
                    model=model,
                    tokenizer=self.tokenizer,
                    examples=self.residual_train_examples,
                    layer_idx=self.layer_idx,
                )
                test_residuals = collect_residuals_per_edge_group(
                    model=model,
                    tokenizer=self.tokenizer,
                    examples=self.eval_test_examples,
                    layer_idx=self.layer_idx,
                )
            model.train()
            append_eval_result(self.eval_file, result)
            if self.collect_residuals_flag:
                append_residual_step(
                    self.residuals_file,
                    step=step,
                    repeat_id=repeat_id,
                    layer_idx=self.layer_idx,
                    train_residuals=train_residuals,
                    test_residuals=test_residuals,
                )
            return control

    residuals_dir = run_dir / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    residuals_file = residuals_dir / "pca_residuals.json"

    torch = importlib.import_module("torch")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_lora:
        if config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = _load_model(
                quantization_config=bnb_config,
                attn_implementation=config.attn_implementation,
                dtype=torch.bfloat16,
            )
        else:
            model = _load_model(
                attn_implementation=config.attn_implementation,
                dtype=torch.bfloat16,
            )
        if config.gradient_checkpointing:
            model.enable_input_require_grads()
        _default_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules or _default_lora_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        model = _load_model(
            attn_implementation=config.attn_implementation,
            dtype=torch.bfloat16,
        )
        if config.freeze_embeddings:
            n_frozen = 0
            try:
                emb = model.get_input_embeddings()
                for p in emb.parameters():
                    p.requires_grad_(False); n_frozen += p.numel()
            except Exception:
                pass
            try:
                head = model.get_output_embeddings()
                if head is not None:
                    for p in head.parameters():
                        p.requires_grad_(False); n_frozen += p.numel()
            except Exception:
                pass
            print(f"[freeze_embeddings] froze {n_frozen/1e6:.1f}M params (input embed + lm_head)")

    # Residual analysis setup: determine target layer and sample ≤8 train edges
    n_layers = model.config.num_hidden_layers
    layer_idx = residual_layer_idx(n_layers)
    rng_edges = random.Random(config.seed + repeat_id)
    all_train_edges: list[Edge] = sorted(
        {example.edge for example in repeat_eval_train_examples}
    )
    if len(all_train_edges) > 8:
        selected_train_edges = set(rng_edges.sample(all_train_edges, 8))
    else:
        selected_train_edges = set(all_train_edges)
    residual_train_examples = [
        ex for ex in repeat_eval_train_examples if ex.edge in selected_train_edges
    ]

    train_dataset = _build_randomized_training_dataset(
        examples=repeat_train_examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        grad_accum=config.grad_accum,
        seed=config.seed + repeat_id,
        mixin_jsonl=config.mixin_jsonl,
        mixin_ratio=config.mixin_ratio,
        chat_format=config.chat_format,
        system_prompt=config.system_prompt,
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer_tmp"),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=1,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        lr_scheduler_type="cosine",
        optim=(config.optim_override if config.optim_override else ("paged_adamw_8bit" if (config.load_in_4bit or config.paged_adamw_8bit) else "adamw_torch_fused")),
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_strategy="no",
        seed=config.seed + repeat_id,
        report_to="none",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        # bf16=True in TrainingArguments enables mixed-precision AMP, which adds fp32
        # grads + fp32 master weights — fine at small scale, OOMs 14B+ full-param.
        # We want "pure bf16" (model already bf16, no AMP) whenever we're using a
        # bnb 8-bit optimizer of any kind. Those optimizers don't keep fp32 master,
        # and pure bf16 throughout is the standard recipe for memory-tight runs.
        bf16=not (config.paged_adamw_8bit
                  or config.optim_override.endswith("8bit")),
        tf32=True,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    eval_file = run_dir / "eval_results.json"
    repeat_seed = config.seed + repeat_id
    periodic_eval_callback = _PeriodicEvalCallback(
        eval_file=eval_file,
        residuals_file=residuals_file,
        tokenizer=tokenizer,
        eval_train_examples=repeat_eval_train_examples,
        eval_test_examples=repeat_eval_test_examples,
        residual_train_examples=residual_train_examples,
        seed=repeat_seed,
        eval_every=config.eval_every,
        max_steps=config.max_steps,
        layer_idx=layer_idx,
        dense_early_evals=config.dense_early_evals,
        collect_residuals=config.collect_residuals,
        eval_subsample=config.eval_subsample,
        chat_format=config.chat_format,
        system_prompt=config.system_prompt,
    )

    trainer_cls = Trainer
    if config.l2_sp_lambda > 0:
        from .regularizers import make_l2sp_trainer
        trainer_cls = make_l2sp_trainer(base_trainer_cls=Trainer, l2_sp_lambda=config.l2_sp_lambda)

    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
        callbacks=[periodic_eval_callback],
    )

    trainer.model.eval()
    initial_result = evaluate_step(
        model=trainer.model,
        tokenizer=tokenizer,
        repeat_id=repeat_id,
        step=0,
        eval_train_examples=repeat_eval_train_examples,
        eval_test_examples=repeat_eval_test_examples,
        seed=repeat_seed,
        chat_format=config.chat_format,
        system_prompt=config.system_prompt,
    )
    if config.collect_residuals:
        initial_train_residuals = collect_residuals_per_edge_group(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=residual_train_examples,
            layer_idx=layer_idx,
        )
        initial_test_residuals = collect_residuals_per_edge_group(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=repeat_eval_test_examples,
            layer_idx=layer_idx,
        )
    trainer.model.train()
    append_eval_result(eval_file, initial_result)
    if config.collect_residuals:
        append_residual_step(
            residuals_file,
            step=0,
            repeat_id=repeat_id,
            layer_idx=layer_idx,
            train_residuals=initial_train_residuals,
            test_residuals=initial_test_residuals,
        )

    trainer.train()

    # ── Save the fine-tuned model. Defaulting this ON because otherwise the
    #    weights live only in VRAM and are lost the instant the pod stops.
    #    Overridable via TrainingConfig.save_final = False (rarely useful). ──
    if config.save_final:
        final_dir = run_dir / "final"
        try:
            # For LoRA the trained object is a PeftModel; trainer.save_model would
            # write only the adapter files. The decisiveness eval loads final/ with
            # a plain AutoModelForCausalLM.from_pretrained, which cannot apply a bare
            # adapter — so merge the LoRA delta into the base weights and save a
            # standalone full model (scored for BOTH accuracy and decisiveness).
            if config.use_lora:
                merged = trainer.model.merge_and_unload()
                merged.save_pretrained(str(final_dir))
            else:
                trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))
            # Small run-provenance dump alongside the weights.
            import json as _json
            import subprocess as _sp
            def _git(*args: str) -> str:
                try:
                    return _sp.check_output(["git", *args], cwd=str(Path(__file__).resolve().parent),
                                            stderr=_sp.DEVNULL, text=True).strip()
                except Exception:
                    return "unknown"
            git_sha = _git("rev-parse", "HEAD")
            git_dirty = _git("status", "--porcelain") not in ("", "unknown")
            provenance = {
                "git_commit": git_sha,
                "git_dirty": git_dirty,  # True => uncommitted changes at run time (should be False!)
                "model_name": config.model_name,
                "max_steps": config.max_steps,
                "lr": config.lr,
                "l2_sp_lambda": config.l2_sp_lambda,
                "batch_size": config.batch_size,
                "grad_accum": config.grad_accum,
                "chat_format": config.chat_format,
                "system_prompt": config.system_prompt,
                "freeze_embeddings": config.freeze_embeddings,
                "use_lora": config.use_lora,
                "lora_r": config.lora_r,
                "seed": config.seed + repeat_id,
                "output_dir": str(run_dir),
            }
            (final_dir / "training_provenance.json").write_text(_json.dumps(provenance, indent=2))
            print(f"[save_final] wrote {final_dir}")
        except Exception as exc:  # pragma: no cover
            print(f"[save_final] FAILED: {exc!r}")

        if config.hf_repo_id:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo_id=config.hf_repo_id, repo_type="model",
                                private=config.hf_private, exist_ok=True)
                api.upload_folder(
                    folder_path=str(final_dir),
                    repo_id=config.hf_repo_id,
                    repo_type="model",
                    commit_message=f"upload from {run_dir.name}",
                )
                print(f"[hf_upload] pushed to https://huggingface.co/{config.hf_repo_id}")
            except Exception as exc:  # pragma: no cover
                print(f"[hf_upload] FAILED: {exc!r}. Model still on disk at {final_dir}.")
