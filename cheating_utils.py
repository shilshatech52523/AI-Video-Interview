import cv2
import face_recognition

# Reference encoding (candidate face)
reference_encoding = None
cheating_frame_count = 0
device_frame_count = 0
required_consecutive_frames = 3

def set_reference_face(frame):
    global reference_encoding
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        reference_encoding = encodings[0]
        return True
    return False

def detect_cheating(frame):
    global cheating_frame_count, device_frame_count

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if not encodings or reference_encoding is None:
        return False

    matches = face_recognition.compare_faces([reference_encoding], encodings[0])
    if not matches[0]:  
        cheating_frame_count += 1
    else:
        cheating_frame_count = 0

    return cheating_frame_count >= required_consecutive_frames

def reset_cheating():
    global cheating_frame_count, device_frame_count, reference_encoding
    cheating_frame_count = 0
    device_frame_count = 0
    reference_encoding = None
