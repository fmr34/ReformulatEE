"""
Fase 3 — Treinamento DPO com curriculo de alpha-annealing.

Arquitetura:
  - Policy model: GPT-2 (124M) para demo CPU  |  Mistral/Llama para GPU
  - LoRA (PEFT) para fine-tuning eficiente
  - DPOTrainer (TRL) com beta=0.1 (temperatura da divergencia KL)
  - Curriculo: treina tier1 -> tier2 -> tier3 em sequencia
  - Stage 1 filter: rejeita samples onde EE(chosen) <= EE(q_bad) + epsilon

Configuracao via .env:
  DPO_MODEL   = gpt2  (ou microsoft/phi-2, meta-llama/Llama-3.2-3B-Instruct, etc.)
  DPO_EPOCHS  = 2     (por tier)
  DPO_BATCH   = 2
  DPO_LR      = 5e-5
  DPO_BETA    = 0.1
  LORA_R      = 8
  LORA_ALPHA  = 16

Uso:
  .venv\\Scripts\\python -m src.rl.train_dpo

Para GPU (adicionar ao .env):
  DPO_MODEL=meta-llama/Llama-3.2-3B-Instruct
  DPO_EPOCHS=3
  DPO_BATCH=8
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuracao (lida do .env com defaults para CPU)
# ---------------------------------------------------------------------------

DPO_MODEL = os.getenv("DPO_MODEL", "gpt2")
DPO_EPOCHS = int(os.getenv("DPO_EPOCHS", "2"))
DPO_BATCH = int(os.getenv("DPO_BATCH", "2"))
DPO_LR = float(os.getenv("DPO_LR", "5e-5"))
DPO_BETA = float(os.getenv("DPO_BETA", "0.1"))
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
MAX_LENGTH = int(os.getenv("DPO_MAX_LEN", "256"))
OUT_DIR = Path(os.getenv("DPO_OUT_DIR", "data/models/dpo_policy"))
DATA_DIR = Path("data/rl")

TIERS = [1, 2, 3]


def pr(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Carregamento do dataset por tier
# ---------------------------------------------------------------------------


def _load_tier(tier: int) -> list[dict]:
    path = DATA_DIR / f"dpo_tier{tier}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _to_hf_dataset(samples: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "prompt": s["prompt"],
                "chosen": s["chosen"],
                "rejected": s["rejected"],
            }
            for s in samples
        ]
    )


# ---------------------------------------------------------------------------
# Configuracao do modelo e LoRA
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    pr(f"  Carregando modelo: {DPO_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DPO_MODEL)

    # GPT-2 nao tem pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        DPO_MODEL,
        device_map=device_map,
        dtype="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    pr(f"  Parametros: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


def _lora_config():
    from peft import LoraConfig
    from peft import TaskType

    # Modulos alvo dependem da arquitetura
    if "gpt2" in DPO_MODEL.lower():
        target = ["c_attn", "c_proj"]
    elif "llama" in DPO_MODEL.lower() or "mistral" in DPO_MODEL.lower():
        target = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "phi" in DPO_MODEL.lower():
        target = ["q_proj", "k_proj", "v_proj", "dense"]
    else:
        target = None  # auto-detect

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Treino DPO por tier (curriculo)
# ---------------------------------------------------------------------------


def _treinar_tier(
    tier: int,
    model,
    tokenizer,
    peft_config,
    checkpoint_dir: Path,
) -> None:
    import torch
    from trl import DPOConfig
    from trl import DPOTrainer

    samples = _load_tier(tier)
    if not samples:
        pr(f"  Tier {tier}: sem dados, pulando.")
        return

    alpha = samples[0]["alpha"]
    pr(f"\n  --- Tier {tier} (alpha={alpha}) | {len(samples)} amostras ---")

    dataset = _to_hf_dataset(samples)
    tier_out = checkpoint_dir / f"tier{tier}"
    tier_out.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    args = DPOConfig(
        output_dir=str(tier_out),
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_BATCH,
        gradient_accumulation_steps=4 if not use_gpu else 1,
        learning_rate=DPO_LR,
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=not use_gpu,
        bf16=use_gpu,
        fp16=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    pr(f"  Iniciando treino tier {tier}...")
    trainer.train()
    trainer.save_model(str(tier_out / "final"))
    pr(f"  Modelo tier {tier} salvo em: {tier_out / 'final'}")


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------


def treinar() -> None:
    pr("=" * 60)
    pr("  Fase 3 — DPO Training com curriculo alpha-annealing")
    pr("=" * 60)
    pr(f"\n  Modelo   : {DPO_MODEL}")
    pr(f"  Epochs   : {DPO_EPOCHS} por tier")
    pr(f"  Batch    : {DPO_BATCH}")
    pr(f"  LR       : {DPO_LR}")
    pr(f"  DPO beta : {DPO_BETA}")
    pr(f"  LoRA r   : {LORA_R}, alpha={LORA_ALPHA}")
    pr(f"  Max len  : {MAX_LENGTH}")
    pr(f"  Out      : {OUT_DIR}")

    import torch

    pr(f"  Device   : {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Carrega modelo base e LoRA
    pr("\n[1/2] Carregando modelo e configurando LoRA...")
    model, tokenizer = _load_model_and_tokenizer()
    peft_config = _lora_config()

    # Treino curriculo: tier1 -> tier2 -> tier3
    pr("\n[2/2] Treino em curriculo (tier 1 -> 2 -> 3)...")
    for tier in TIERS:
        _treinar_tier(tier, model, tokenizer, peft_config, OUT_DIR)

    # Salva tokenizer e config final
    final_dir = OUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(final_dir))

    pr(f"\n{'='*60}")
    pr("  Treinamento concluido!")
    pr(f"  Modelo final: {OUT_DIR / 'tier3' / 'final'}")
    pr(f"  Tokenizer  : {final_dir}")
    pr(f"{'='*60}")


# ---------------------------------------------------------------------------
# Demo rapida (smoke test) — roda 5 steps em CPU
# ---------------------------------------------------------------------------


def smoke_test() -> None:
    """
    Executa apenas 5 steps de treino no Tier 1 para validar o pipeline.
    Util para verificar se o codigo funciona antes de um treino completo.
    """
    import os as _os

    _os.environ["DPO_EPOCHS"] = "1"

    pr("=" * 60)
    pr("  Smoke test DPO (5 steps, Tier 1 apenas)")
    pr("=" * 60)

    from trl import DPOConfig
    from trl import DPOTrainer

    samples = _load_tier(1)[:20]  # apenas 20 amostras
    dataset = _to_hf_dataset(samples)

    model, tokenizer = _load_model_and_tokenizer()
    peft_config = _lora_config()

    import torch as _torch

    args = DPOConfig(
        output_dir=str(OUT_DIR / "smoke_test"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        beta=0.1,
        max_length=128,
        max_steps=5,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=not _torch.cuda.is_available(),
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    pr("\n  Executando 5 steps...")
    trainer.train()
    pr("  Smoke test OK!")
    pr(f"  Checkpoints em: {OUT_DIR / 'smoke_test'}")


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        smoke_test()
    else:
        treinar()
