import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

MODEL = "HuggingFaceTB/SmolLM2-135M"

# Free up GPU memory
torch.cuda.empty_cache()

dataset = load_dataset("NeelNanda/pile-10k", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)

gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.float16,
)
gpt.gradient_checkpointing_enable()  # Optional but helpful

cfg = TrainConfig(SaeConfig(), batch_size=1)  # VERY IMPORTANT
trainer = Trainer(cfg, tokenized, gpt)

trainer.fit()
