from fastapi import FastAPI
import json
import uvicorn
from utils.prod import SimpleT5

CONFIG_PATH = "./app/sidecard/config.json"

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


if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0", port=5000) 


