import cv2 
import os
import websockets
import numpy as np
import time
import asyncio

from dotenv import load_dotenv

load_dotenv()

WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
FPS_TARGET = int(os.getenv("FPS_TARGET", 30))
FRAME_DELAY = 1.0 / FPS_TARGET


async def video_client():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(  f"Connecting to WebSocket server at {WEBSOCKET_URL}...")

    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            print("Connected to WebSocket server.")

            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from video device.")
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send(buffer.tobytes())

                response = await websocket.recv()
                nparr = np.frombuffer(response, np.uint8)
                echoed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.imshow('Original Video', frame)
                cv2.imshow('Echoed Video', echoed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # F. Frame Rate Control (Backpressure Lite)
                # Ensures we don't spam the network too fast
                elapsed_time = time.time() - start_time
                wait_time = FRAME_DELAY - elapsed_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

    except Exception as e:
        print(f"Error: Could not connect to WebSocket server: {e}")
        cap.release()
        return
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video capture ended and resources released.")

# --- Execute Client ---
if __name__ == "__main__":
    # Windows-specific fix for async/await
    if os.name == 'nt':
        import os
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
    asyncio.run(video_client())