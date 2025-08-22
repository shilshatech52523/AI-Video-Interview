import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO




cap = cv2.VideoCapture("video.mp4")


# ----------------- Global Vars -----------------
reference_encoding = None
cheating_frame_count = 0
device_frame_count = 0
required_consecutive_frames = 3

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # use small model for speed

def analyze_frame(frame):
    global cheating_frame_count, device_frame_count

    if reference_encoding is None:
        return "Reference face not set"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if not encodings:
        cheating_frame_count += 1
        return "No face detected"
    
    # --- Face match check ---
    match = face_recognition.compare_faces([reference_encoding], encodings[0])[0]
    if not match:
        cheating_frame_count += 1
        return "Different person detected"

    # --- Device detection (YOLO every Nth frame, not random) ---
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # current frame number
    if frame_id % 10 == 0:  # check every 10th frame (more reliable than random)
        results = yolo_model(frame, verbose=False)
        for r in results:
            for cls, conf in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                cls = int(cls)
                if cls in [63, 64, 65, 67]:  
                    # 63 = laptop, 64 = mouse, 65 = remote, 67 = cell phone
                    if conf > 0.5:  # only if confidence > 50%
                        device_frame_count += 1
                        return "Device detected"

    return "Normal"

# ----------------- Face Setup -----------------
def set_reference_face(frame):
    global reference_encoding
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        reference_encoding = encodings[0]
        return True
    return False


def check_face(frame):
    global reference_encoding, cheating_frame_count
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if not encodings:
        return "⚠ No face detected"

    matches = face_recognition.compare_faces([reference_encoding], encodings[0])
    if not matches[0]:
        cheating_frame_count += 1
        if cheating_frame_count >= required_consecutive_frames:
            return "🚨 Face mismatch detected"
    else:
        cheating_frame_count = 0
    return None


# ----------------- YOLO Object Detection -----------------
def check_objects(frame):
    global device_frame_count

    if np.random.randint(0, 30) == 1:  # YOLO runs only sometimes
        results = yolo_model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = yolo_model.names[cls]

                # Example: detect mobile or extra persons
                if label.lower() in ["cell phone", "laptop"]:
                    device_frame_count += 1
                    if device_frame_count >= required_consecutive_frames:
                        return f"🚨 Device detected: {label}"
                elif label.lower() in ["person"]:
                    # multiple persons
                    if len(result.boxes) > 1:
                        return "🚨 Multiple persons detected"

    return None


# ----------------- Main Loop -----------------
def start_proctoring():
    cap = cv2.VideoCapture(0)
    print("Press 's' to set reference face, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Proctoring System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if set_reference_face(frame):
                print("✅ Reference face set")
            else:
                print("⚠ No face found")

        if reference_encoding is not None:
            # Face check
            face_status = check_face(frame)
            if face_status:
                print(face_status)

            # Object check
            object_status = check_objects(frame)
            if object_status:
                print(object_status)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_proctoring()
