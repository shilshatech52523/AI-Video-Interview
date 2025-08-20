# websocket_routes.py
import json, time, base64, cv2, numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from screenshot_utils import save_ws_screenshot
from questions_utils import sessions_state, extra_person_count, get_next_question
from ultralytics import YOLO
import mediapipe as mp

# Mediapipe FaceMesh (multi-face)
mp_face_mesh = mp.solutions.face_mesh
face_mesh_ws = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)

# YOLO model
yolo_model = YOLO('yolov8n.pt')  # keep file in project root


async def websocket_questions(ws: WebSocket):
    await ws.accept()
    session_id = str(int(time.time() * 1000))
    extra_person_count[session_id] = 0
    sessions_state[session_id] = {"question_index": 0, "active_question": ""}

    try:
        # send session_id to client
        await ws.send_text(json.dumps({"type": "session", "session_id": session_id}))

        while True:
            msg = await ws.receive_text()

            # JSON parse / backward compat
            try:
                data = json.loads(msg)
            except Exception:
                if msg.startswith("frame:"):
                    data = {"type": "frame", "data": msg.split(":", 1)[1]}
                elif msg == "next_question":
                    data = {"type": "next_question"}
                else:
                    data = {"type": "unknown"}

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

                # Face checks
                if len(faces) == 0:
                    await ws.send_text(json.dumps({"type": "warning", "message": "No face detected"}))
                elif len(faces) > 1:
                    extra_person_count[session_id] += 1
                    await ws.send_text(json.dumps({"type": "warning", "message": f"Multiple faces! Count={extra_person_count[session_id]}"}))
                    if extra_person_count[session_id] >= 3:
                        save_ws_screenshot(frame, session_id)
                        await ws.send_text(json.dumps({"type": "stop", "message": "Cheating detected! Session closed."}))
                        break
                else:
                    extra_person_count[session_id] = 0

                # Random screenshot
                if np.random.randint(0, 60) == 1:
                    save_ws_screenshot(frame, session_id)

                # Device detection (random interval)
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
                            await ws.send_text(json.dumps({"type": "stop", "message": "Device detected! Session closed."}))
                            break
                    except Exception:
                        pass

            elif data.get("type") == "next_question":
                q_data = await get_next_question(session_id)
                await ws.send_text(json.dumps(q_data))
                if q_data["type"] == "stop":
                    break

            elif data.get("type") == "stop":
                await ws.send_text(json.dumps({"type": "stop", "message": "Session Completed"}))
                break

    except WebSocketDisconnect:
        print("Client disconnected from /ws")
    except Exception as e:
        print("WebSocket /ws error:", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass
