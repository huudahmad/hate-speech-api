from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#Pydantic base model to define the structure of incoming requests (strings)
class AnalysisRequest(BaseModel):
    text: str

MODEL_PATH = "./hate_speech_model"

#load model once at beginning
print("Loading model")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)                       #tokenizer object loaded past on tokenizer from model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)      #model object loaded, instance of trained model
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

#create FastAPI instance
app = FastAPI(title="Hate Speech Detection API")

@app.post("/predict")
async def predict(request: AnalysisRequest):
    #Step 1: preprocess the text (Tokenize)
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", #need to return PyTorch tensors
        truncation=True, 
        padding=True, 
        max_length=512
    )

    # Step 2: feed it to the model
    with torch.no_grad(): #no need to calculate gradients for inference
        outputs = model(**inputs)

    # Step 3:Interpret the results
    probs = F.softmax(outputs.logits, dim=1)
    
    #Get the highest probability (Winner)
    #labels: 0 = Not Hate, 1 = Hate (based on your previous training mapping)
    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    
    label_map = {0: "NORMAL", 1: "HATE"} 
    
    return {
        "label": label_map[pred_idx],
        "confidence": confidence,
        "input_text": request.text
    }

#simple GET endpoint to see if the server is alive
@app.get("/health")
def health_check():
    return {"status": "operational"}