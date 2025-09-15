from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import os

# Routers / Utils
from utils.pdf_exporter import router as pdf_exporter
from utils.upload_router import router as upload_router
from utils.ws_socket import router as ws_socket_router
from utils.generate_camera import generate_camera_frames, stop_camera_flag
from utils.cheating_detection import start_proctoring

# ------------------------------------------------
# App Initialization
# ------------------------------------------------
app = FastAPI()
router = APIRouter()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("pdf", exist_ok=True)
os.makedirs("save_screenshot", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Static mounts
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")
app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/pdfs", StaticFiles(directory="pdf"), name="pdfs")  # optional static serving
app.mount("/screens", StaticFiles(directory="save_screenshot"), name="screens")

# Templates
templates = Jinja2Templates(directory="templates")

# ------------------------------------------------
# Routers
# ------------------------------------------------
app.include_router(upload_router)
app.include_router(ws_socket_router)
app.include_router(pdf_exporter)

# ------------------------------------------------
# Routes
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/start_proctoring")
def run_proctoring():
    start_proctoring()
    return {"status": "started"}


@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/stop_camera")
def stop_camera():
    stop_camera_flag.set()
    return {"message": "Camera stopped"}
