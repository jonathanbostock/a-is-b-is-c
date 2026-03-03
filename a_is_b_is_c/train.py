from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
import random
from typing import Any

from datasets import Dataset

from .dataset import PromptExample
from .evaluate import append_eval_result, evaluate_step


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
    batch_size: int = 8
    grad_accum: int = 2
    lora_r: int = 16
    seed: int = 42
    output_dir: str = "./runs"
    model_name: str = "google/gemma-3-1b-pt"
    max_seq_length: int = 256


def _build_label_masked_record(*, tokenizer: Any, prompt: str, completion: str, max_seq_length: int) -> dict[str, Any]:
    full_text = f"{prompt} {completion}".strip()
    encoded_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
    )
    encoded_prompt = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=True,
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


def _build_randomized_training_dataset(
    *,
    examples: list[PromptExample],
    tokenizer: Any,
    max_seq_length: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    seed: int,
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

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(total_samples):
        edge = rng.choice(edges)
        template = rng.choice(templates_by_edge[edge])
        selected = rng.choice(grouped[edge][template])
        rows.append(
            _build_label_masked_record(
                tokenizer=tokenizer,
                prompt=selected.prompt,
                completion=selected.completion,
                max_seq_length=max_seq_length,
            )
        )

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
    BitsAndBytesConfig = transformers_module.BitsAndBytesConfig
    DataCollatorForSeq2Seq = transformers_module.DataCollatorForSeq2Seq
    Trainer = transformers_module.Trainer
    TrainerCallback = transformers_module.TrainerCallback
    TrainingArguments = transformers_module.TrainingArguments
    AutoLigerKernelForCausalLM = liger_module.AutoLigerKernelForCausalLM
    LoraConfig = peft_module.LoraConfig
    TaskType = peft_module.TaskType
    get_peft_model = peft_module.get_peft_model

    class _PeriodicEvalCallback(TrainerCallback):
        def __init__(
            self,
            *,
            eval_file: Path,
            tokenizer: Any,
            eval_train_examples: list[PromptExample],
            eval_test_examples: list[PromptExample],
            seed: int,
            eval_every: int,
        ) -> None:
            self.eval_file = eval_file
            self.tokenizer = tokenizer
            self.eval_train_examples = eval_train_examples
            self.eval_test_examples = eval_test_examples
            self.seed = seed
            self.eval_every = eval_every

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(state.global_step)
            if step <= 0 or step % self.eval_every != 0:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            result = evaluate_step(
                model=model,
                tokenizer=self.tokenizer,
                repeat_id=repeat_id,
                step=step,
                eval_train_examples=self.eval_train_examples,
                eval_test_examples=self.eval_test_examples,
                seed=self.seed,
            )
            append_eval_result(self.eval_file, result)
            return control

    repeat_dir = run_dir / "checkpoints" / f"repeat_{repeat_id}"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    torch = importlib.import_module("torch")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoLigerKernelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    train_dataset = _build_randomized_training_dataset(
        examples=repeat_train_examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        grad_accum=config.grad_accum,
        seed=config.seed + repeat_id,
    )

    training_args = TrainingArguments(
        output_dir=str(repeat_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=1,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=10,
        save_steps=config.eval_every,
        seed=config.seed + repeat_id,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    eval_file = run_dir / "eval_results.json"
    repeat_seed = config.seed + repeat_id
    periodic_eval_callback = _PeriodicEvalCallback(
        eval_file=eval_file,
        tokenizer=tokenizer,
        eval_train_examples=repeat_eval_train_examples,
        eval_test_examples=repeat_eval_test_examples,
        seed=repeat_seed,
        eval_every=config.eval_every,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
        callbacks=[periodic_eval_callback],
    )

    initial_result = evaluate_step(
        model=trainer.model,
        tokenizer=tokenizer,
        repeat_id=repeat_id,
        step=0,
        eval_train_examples=repeat_eval_train_examples,
        eval_test_examples=repeat_eval_test_examples,
        seed=repeat_seed,
    )
    append_eval_result(eval_file, initial_result)

    trainer.train()
