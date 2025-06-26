from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="fakespot-ai/roberta-base-ai-text-detection-v1",
    tokenizer="fakespot-ai/roberta-base-ai-text-detection-v1"
)

def roberta_detect(text: str):
    result = classifier(text[:512])[0]  # limit length for token safety
    return {
        "label": result["label"],
        "confidence": round(result["score"] * 100, 2)
    }
