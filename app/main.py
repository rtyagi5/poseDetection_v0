from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import threading
from app.models.test_project import start_detection

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_detection")
async def start_detection_endpoint(left_reps: int = Form(...), right_reps: int = Form(...), rest_duration: int = Form(...)):
    detection_thread = threading.Thread(target=start_detection, args=(left_reps, right_reps, rest_duration))
    detection_thread.start()
    return JSONResponse(content={"status": "Detection started"})

@app.get("/status/")
async def get_status():
    return JSONResponse(content={"status": "Pose detection is running"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
