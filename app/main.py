from fastapi import FastAPI
import json
import uvicorn
from utils.prod import SimpleT5
from google.cloud import storage
import zipfile
import os

CONFIG_PATH = "./app/sidecards/config.json"
BUCKET = "sagewrite"
credentials_dir = "./app/sidecards/sagewrite.json"
name = "model.zip"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_dir

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET)
blob = bucket.blob(str(name))
blob.download_to_filename(str(name))

with zipfile.ZipFile("model.zip", 'r') as zip_ref:
    zip_ref.extractall("model")
    
with open(CONFIG_PATH) as f:
    config = json.load(f)

model = SimpleT5()

model.load_model(
    **config["model"]
                )

app = FastAPI(
    **config["api"]
)


@app.post("/generate")
async def post_test(text: str):
    return model.predict(text)


#if __name__ == "__main__":
#    uvicorn.run("main:app",host="0.0.0.0", port=5000) 


