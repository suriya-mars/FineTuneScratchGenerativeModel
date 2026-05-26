"""Model loading via Unsloth — 4-bit QLoRA for 6GB VRAM."""
from unsloth import FastLanguageModel


def load_model_and_tokenizer(config: dict):
    model_cfg = config["model"]
    lora_cfg  = config["lora"]
    train_cfg = config["training"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_len"],
        dtype=None,       # auto-detect: float16 on RTX 3050
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM than HF default
        random_state=train_cfg["seed"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
