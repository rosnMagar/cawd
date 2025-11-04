from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_name="yolov8n.pt"):
        try:
            print(f"Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def predict(self, frame):    
        res = self.model.predict(
            source=frame, 
            imgsz=640, 
            conf=0.25,
            verbose=False
        )
        return res
        
    def annotate(self, frame):
        annotated_frame = self.predict(frame)[0].plot()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        ret_encode, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)

        if not ret_encode:
            print("Error encoding annotated frame.")
            return None

        return buffer.tobytes()


