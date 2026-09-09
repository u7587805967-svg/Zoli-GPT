from ragas import evaluate
from datasets import Dataset

def run_evals(test_dataset: Dataset):
    results = evaluate(test_dataset)
    print(f"RAGas teszt pontszámok: {results}")

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)