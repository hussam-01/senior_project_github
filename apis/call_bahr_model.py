from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pydantic import BaseModel
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Item(BaseModel):
    text : str

app = FastAPI()

# Load model
model_name = "../models/poetry_classification_models/classify_by_bahr/arapoems_datatset_fine_tuning_akhooli_bahr_classification/best"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Configure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

def predict_one(text):
    """Predict class for a single text"""
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
    
    predicted_class = prediction.item()
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

@app.post("/get_bahr/")
async def read_root(item: Item):
    # text = "وَلَو أَنّي أَشاءُ لَقُلتُ لَهُ حَبيبي أَنتَ مِن عَبدٍ سَليمِ ِذاً لَقُلتُ لَهُ اِستَمع مِنهُ قَولاً يَكونُ بِهِ عَل"
    text = item.text
    pred_class, confidence = predict_one(text)
    # print(f"Predicted class: {pred_class}")
    # print(f"Confidence: {confidence:.4f}")
    indexes = ['rajaz', 'khafif', 'baseet', 'taweel', 'wafer', 'kamel', 'saree',
       'mutaqarib', 'ramel', 'hazaj', 'madeed', 'munsarih', 'mujtath',
       'mutadarak', 'muashah', 'colloquial']
    return {"output": str(indexes[pred_class]),"confidence":str(confidence)}