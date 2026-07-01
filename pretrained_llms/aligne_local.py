"""Run aligne's decisiveness panel against a local HF model, no HTTP/vLLM.

A LocalChatClient satisfies the aligne ChatClient duck-type (async chat()) by
running a forward pass through a transformers model and returning top-token
logprobs in OpenAI chat-completions format. This reuses aligne's exact Case-V
fit + decisiveness_fitted math while sidestepping the vLLM CUDA-13 wheel issue.
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aligne.metrics.preferences import run_panel, PanelConfig

DATA_CONCEPTS = Path("/root/aligne/src/aligne/data/concepts.json")


class LocalChatClient:
    """Duck-typed stand-in for aligne.client.ChatClient."""
    def __init__(self, model_path: str, device: str = "cuda", tokenizer_path: str = None):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        self.device = device

    async def aclose(self):
        pass

    async def chat(self, payload: dict) -> dict:
        msgs = payload["messages"]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        want_logprobs = bool(payload.get("logprobs"))
        with torch.no_grad():
            if want_logprobs:
                logits = self.model(**enc).logits[0, -1].float()
                lp = torch.log_softmax(logits, dim=-1)
                k = int(payload.get("top_logprobs", 20))
                top = torch.topk(lp, k)
                tl = [{"token": self.tok.decode([tid]), "logprob": float(v)}
                      for tid, v in zip(top.indices.tolist(), top.values.tolist())]
                first_tok = self.tok.decode([top.indices[0].item()])
                return {"choices": [{
                    "message": {"content": first_tok},
                    "logprobs": {"content": [{"top_logprobs": tl}]},
                }]}
            else:
                n = int(payload.get("n", 1))
                temp = float(payload.get("temperature", 1.0))
                gen = self.model.generate(
                    **enc, max_new_tokens=int(payload.get("max_tokens", 8)),
                    do_sample=temp > 0, temperature=max(temp, 0.01),
                    num_return_sequences=n, pad_token_id=self.tok.pad_token_id,
                )
                out = []
                for g in gen:
                    txt = self.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    out.append({"message": {"content": txt}})
                return {"choices": out}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-concepts", type=int, default=155)
    args = ap.parse_args()

    client = LocalChatClient(args.model)
    cfg = PanelConfig(seed=args.seed, n_concepts=args.n_concepts,
                      concepts_path=DATA_CONCEPTS)
    panel = await run_panel(client, cfg, Path(args.out))
    print("PANEL_RESULT", json.dumps(panel))


if __name__ == "__main__":
    asyncio.run(main())
