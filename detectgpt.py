from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random

MODEL_NAME = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

def get_log_probs(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
    return log_likelihood

def perturb_text(text, num_samples=5):
    words = text.split()
    perturbed_texts = []
    for _ in range(num_samples):
        idx = random.randint(0, len(words)-1)
        perturbed = words[:idx] + [random.choice(words)] + words[idx+1:]
        perturbed_texts.append(" ".join(perturbed))
    return perturbed_texts

def detect_gpt_score(text):
    base_score = get_log_probs(text)
    perturbed_scores = [get_log_probs(pt) for pt in perturb_text(text)]
    mean_diff = np.mean([base_score - ps for ps in perturbed_scores])
    return round(mean_diff, 4)
