from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from datasets import Dataset

from .dataset import PromptExample
from .evaluate import append_eval_result, evaluate_step


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
    model_name: str = "unsloth/gemma-2-2b"
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


def _examples_to_dataset(*, examples: list[PromptExample], tokenizer: Any, max_seq_length: int) -> Dataset:
    rows = [
        _build_label_masked_record(
            tokenizer=tokenizer,
            prompt=example.prompt,
            completion=example.completion,
            max_seq_length=max_seq_length,
        )
        for example in examples
    ]
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
    try:
        transformers_module = importlib.import_module("transformers")
        trl_module = importlib.import_module("trl")
        unsloth_module = importlib.import_module("unsloth")
    except ImportError as exc:  # pragma: no cover
        msg = (
            "Training dependencies are missing. Install with uv add unsloth transformers trl "
            "datasets accelerate bitsandbytes torch."
        )
        raise RuntimeError(msg) from exc

    DataCollatorForSeq2Seq = transformers_module.DataCollatorForSeq2Seq
    TrainingArguments = transformers_module.TrainingArguments
    SFTTrainer = trl_module.SFTTrainer
    FastLanguageModel = unsloth_module.FastLanguageModel

    repeat_dir = run_dir / "checkpoints" / f"repeat_{repeat_id}"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
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
        use_gradient_checkpointing=True,
        random_state=config.seed + repeat_id,
    )

    train_dataset = _examples_to_dataset(
        examples=repeat_train_examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    training_args = TrainingArguments(
        output_dir=str(repeat_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=10,
        save_steps=config.eval_every,
        seed=config.seed + repeat_id,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        dataset_text_field=None,
    )

    trainer.train()

    eval_file = run_dir / "eval_results.json"
    for step in range(config.eval_every, config.max_steps + 1, config.eval_every):
        checkpoint_path = repeat_dir / f"checkpoint-{step}"
        if checkpoint_path.exists():
            model = trainer.model.from_pretrained(checkpoint_path)

        result = evaluate_step(
            model=trainer.model,
            tokenizer=tokenizer,
            step=step,
            eval_train_examples=repeat_eval_train_examples,
            eval_test_examples=repeat_eval_test_examples,
            seed=config.seed + repeat_id,
        )
        append_eval_result(eval_file, result)
