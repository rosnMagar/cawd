from fastapi import FastAPI, WebSocket
import logging
from ml.detection import Detector
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def read_root():  
    return {"Hello": "World"}

@app.websocket("/ws/{model_name}")
async def websocket_endpoint(websocket: WebSocket, model_name: str):
    await websocket.accept()

    if model_name == "":
        model_name = "yolov8n.pt"
    else:
        if model_name not in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
            model_name = "yolov8n"
        else:
            model_name = f"{model_name}.pt"
    try:
        while True:
            data = await websocket.receive_bytes()


            # 2. DECODE: Convert binary JPEG to OpenCV image (NumPy array)
            nparr = np.frombuffer(data, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            detector = Detector(model_name=model_name)
            annotated_frame = detector.annotate(data)

            await websocket.send_bytes(annotated_frame)
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
        await websocket.close()
    
if __name__ == "__main__":
    logger.info("Starting server with: uvicorn server:app --reload --host 0.0.0.0 --port 8000")