from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from roberta_model import roberta_detect
from detectgpt import detect_gpt_score
from file_utils import extract_text_from_file

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "AI Detector backend is running"}

@app.post("/detect")
def detect_roberta(input: TextInput):
    return roberta_detect(input.text)

@app.post("/detectgpt")
def detect_detectgpt(input: TextInput):
    score = detect_gpt_score(input.text)
    return {"score": score, "label": "AI" if score > 0.2 else "Human"}

@app.post("/upload")
async def upload_and_detect(file: UploadFile = File(...), method: str = Form("roberta")):
    contents = await file.read()
    text = extract_text_from_file(contents, file.filename)

    if method == "roberta":
        return roberta_detect(text)
    elif method == "detectgpt":
        score = detect_gpt_score(text)
        return {"score": score, "label": "AI" if score > 0.2 else "Human"}
    else:
        return {"error": "Invalid method"}

# Optional: for local testing only
if __name__ == "__main__":
    print("This should not run on Render.")
