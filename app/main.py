from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import threading
import base64
import cv2
import numpy as np
from app.models.pose_module import poseDetector
from app.models.test_project import start_detection

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

detector = poseDetector()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_detection")
async def start_detection_endpoint(left_reps: int = Form(...), right_reps: int = Form(...), rest_duration: int = Form(...)):
    detection_thread = threading.Thread(target=start_detection, args=(left_reps, right_reps, rest_duration))
    detection_thread.start()
    return JSONResponse(content={"status": "Detection started"})

@app.post("/process_frame")
async def process_frame(request: Request):
    data = await request.json()
    frame_base64 = data['frame']
    frame_bytes = base64.b64decode(frame_base64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=True)

    _, buffer = cv2.imencode('.jpg', img)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"processed_frame": processed_frame_base64})

@app.get("/status/")
async def get_status():
    return JSONResponse(content={"status": "Pose detection is running"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
