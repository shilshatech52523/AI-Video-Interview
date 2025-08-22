from fastapi import APIRouter, Query, Form, File, UploadFile
from fastapi.responses import FileResponse
from .analysis import extract_frames,  analyze_confidence, analyze_eye_contact, analyze_posture, analyze_smile, answer_quality_score, transcribe_google
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Query, Request, WebSocket, WebSocketDisconnect
import os, shutil, subprocess, json, math, time, random, threading, asyncio, base64
import cv2
import mediapipe as mp
from ultralytics import YOLO
from .screenshot import get_face_bounding_rect, save_ws_screenshot, run_tts
from .question import questions

router = APIRouter()

extra_person_count: dict[str, int] = {}
sessions_state: dict[str, dict] = {}

yolo_model = YOLO('yolov8n.pt')

mp_face_mesh = mp.solutions.face_mesh
face_mesh_analysis = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Mediapipe FaceMesh (WebSocket up to 5 faces)
face_mesh_ws = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)

@router.websocket("/ws")
async def websocket_questions(ws: WebSocket):
    await ws.accept()
    session_id = str(int(time.time() * 1000))
    extra_person_count[session_id] = 0
    sessions_state[session_id] = {"question_index": 0, "active_question": ""}

    try:
        # 🔹 Send session_id immediately to client
        await ws.send_text(json.dumps({"type": "session", "session_id": session_id}))

        while True:
            msg = await ws.receive_text()

            # 🔹 Parse incoming data
            try:
                data = json.loads(msg)
            except Exception:
                if msg.startswith("frame:"):
                    data = {"type": "frame", "data": msg.split(":", 1)[1]}
                elif msg == "next_question":
                    data = {"type": "next_question"}
                else:
                    data = {"type": "unknown"}

            # ---------------- FRAME HANDLING ----------------
            if data.get("type") == "frame":
                encoded_data_url = data.get("data", "")
                if "," in encoded_data_url:
                    encoded = encoded_data_url.split(",", 1)[1]
                else:
                    encoded = encoded_data_url

                img_data = base64.b64decode(encoded)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                results = face_mesh_ws.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                faces = results.multi_face_landmarks or []

                if len(faces) == 1:   # ✅ bounding box draw
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = get_face_bounding_rect(faces[0], w, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    extra_person_count[session_id] = 0  # reset if single face
                elif len(faces) > 1:
                    extra_person_count[session_id] += 1
                    await ws.send_text(json.dumps({
                        "type": "warning",
                        "message": f"Multiple faces! Count={extra_person_count[session_id]}"
                    }))
                    if extra_person_count[session_id] >= 3:
                        save_ws_screenshot(frame, session_id)
                        await ws.send_text(json.dumps({
                            "type": "stop",
                            "message": "Cheating detected! Session closed."
                        }))
                        break
                else:
                    await ws.send_text(json.dumps({
                        "type": "warning",
                        "message": "No face detected"
                    }))

                # ✅ Send frame back to frontend with box
                _, buffer = cv2.imencode('.jpg', frame)
                encoded_frame = base64.b64encode(buffer).decode("utf-8")
                await ws.send_text(json.dumps({
                    "type": "frame_boxed",
                    "data": f"data:image/jpeg;base64,{encoded_frame}"
                }))

                # 🔹 Random screenshot save
                if np.random.randint(0, 60) == 1:
                    save_ws_screenshot(frame, session_id)

                # 🔹 Device detection (YOLO)
                if np.random.randint(0, 10) == 1:
                    try:
                        yolo_results = yolo_model(frame, verbose=False)
                        device_hit = False
                        for r in yolo_results:
                            for box in r.boxes:
                                cls_id = int(box.cls.cpu())
                                conf = float(box.conf.cpu())
                                cls_name = yolo_model.names.get(cls_id, "").lower()
                                if conf > 0.6 and cls_name in ["cell phone", "cellphone", "phone", "laptop", "tv", "tablet"]:
                                    device_hit = True
                                    break
                            if device_hit:
                                break
                        if device_hit:
                            save_ws_screenshot(frame, session_id)
                            await ws.send_text(json.dumps({
                                "type": "stop",
                                "message": "Device detected! Session closed."
                            }))
                            break
                    except Exception:
                        pass

            # ---------------- AUDIO HANDLING ----------------
            elif data.get("type") == "audio":
                try:
                    encoded_audio = data.get("data", "")
                    if "," in encoded_audio:
                        encoded_audio = encoded_audio.split(",", 1)[1]

                    audio_bytes = base64.b64decode(encoded_audio)

                    # Temp save
                    os.makedirs("audio", exist_ok=True)
                    audio_path = f"audio/ws_{session_id}_{int(time.time())}.wav"
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)

                    # ✅ Transcribe
                    transcript = transcribe_google(audio_path)

                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "text": transcript,
                        "session_id": session_id
                    }))

                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"Transcription failed: {str(e)}"
                    }))

            # ---------------- NEXT QUESTION ----------------
            elif data.get("type") == "next_question":
                idx = sessions_state[session_id]["question_index"]
                if idx < len(questions):
                    question = questions[idx]
                    sessions_state[session_id]["question_index"] += 1
                    sessions_state[session_id]["active_question"] = question
                    audio_url = run_tts(question, session_id)
                    await ws.send_text(json.dumps({
                        "type": "question",
                        "text": question,
                        "audio": audio_url,
                        "session_id": session_id,
                        "index": idx
                    }))
                else:
                    await ws.send_text(json.dumps({
                        "type": "stop",
                        "message": "Interview completed"
                    }))
                    break

            # ---------------- STOP ----------------
            elif data.get("type") == "stop":
                await ws.send_text(json.dumps({
                    "type": "stop",
                    "message": "Session Completed"
                }))
                break

            else:
                # ignore unknown
                pass

    except WebSocketDisconnect:
        print("Client disconnected from /ws")
    except Exception as e:
        print("WebSocket /ws error:", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass
