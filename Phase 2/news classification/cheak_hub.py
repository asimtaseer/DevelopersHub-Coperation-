from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "asimtaseer/news-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

labels = model.config.id2label

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_token_type_ids=False
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    if isinstance(labels, dict):
        return labels[prediction]

    return prediction


text = "Pakistan wins cricket match against India"

print("Prediction:", predict(text))