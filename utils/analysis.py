import cv2
import os
import numpy as np
import mediapipe as mp
import math
import librosa
import wave,json
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import speech_recognition as sr

mp_face_mesh = mp.solutions.face_mesh
face_mesh_analysis = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
def extract_frames(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames


def detect_face_landmarks(frames):
    landmarks_list = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_analysis.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks_list.append(results.multi_face_landmarks[0])
        else:
            landmarks_list.append(None)
    return landmarks_list

def euclidean_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calculate_eye_aspect_ratio(landmarks, img_width, img_height):
    left_eye_indices = [33, 133, 159, 145, 153, 154]
    right_eye_indices = [362, 263, 386, 374, 380, 381]
    def eye_ratio(eye_pts):
        A = euclidean_dist((eye_pts[1].x*img_width, eye_pts[1].y*img_height), (eye_pts[5].x*img_width, eye_pts[5].y*img_height))
        B = euclidean_dist((eye_pts[2].x*img_width, eye_pts[2].y*img_height), (eye_pts[4].x*img_width, eye_pts[4].y*img_height))
        C = euclidean_dist((eye_pts[0].x*img_width, eye_pts[0].y*img_height), (eye_pts[3].x*img_width, eye_pts[3].y*img_height))
        return (A + B) / (2.0 * C)
    left_eye = [landmarks.landmark[i] for i in left_eye_indices]
    right_eye = [landmarks.landmark[i] for i in right_eye_indices]
    left_ear = eye_ratio(left_eye)
    right_ear = eye_ratio(right_eye)
    return (left_ear + right_ear) / 2.0

def analyze_eye_contact(landmarks_list, frame_width, frame_height):
    ear_values = []
    for landmarks in landmarks_list:
        if landmarks:
            ear = calculate_eye_aspect_ratio(landmarks, frame_width, frame_height)
            ear_values.append(ear)
    if not ear_values:
        return 0.0
    avg_ear = sum(ear_values) / len(ear_values)
    score = min(max((avg_ear - 0.2) * 5, 0), 1)
    return score

def analyze_posture(landmarks_list):
    tilt_angles = []
    for landmarks in landmarks_list:
        if landmarks:
            nose = landmarks.landmark[1]
            chin = landmarks.landmark[152]
            dx = chin.x - nose.x
            dy = chin.y - nose.y
            angle = math.degrees(math.atan2(dy, dx))
            tilt_angles.append(angle)
    if not tilt_angles:
        return 0.0
    avg_tilt = sum(tilt_angles) / len(tilt_angles)
    score = max(0, 1 - abs(avg_tilt - 90) / 45)
    return score

def analyze_smile(landmarks_list):
    scores = []
    for landmarks in landmarks_list:
        if landmarks:
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            left_corner = landmarks.landmark[61]
            right_corner = landmarks.landmark[291]
            vertical_dist = abs(upper_lip.y - lower_lip.y)
            horizontal_dist = abs(left_corner.x - right_corner.x)
            if horizontal_dist > 0:
                mar = vertical_dist / horizontal_dist
                smile_score = max(0, min(1, 0.3 - mar))
                scores.append(smile_score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def calculate_blink_rate(landmarks_list, fps=30):
    blink_frames = 0
    total_frames = len(landmarks_list)
    EAR_THRESHOLD = 0.2
    for landmarks in landmarks_list:
        if landmarks:
            img_w, img_h = 640, 480
            ear = calculate_eye_aspect_ratio(landmarks, img_w, img_h)
            if ear < EAR_THRESHOLD:
                blink_frames += 1
    blink_rate_per_sec = (blink_frames / total_frames) * fps if total_frames > 0 else 0
    normalized_blink = min(max((blink_rate_per_sec - 0.25) / 0.25, 0), 1)
    blink_score = 1 - abs(normalized_blink - 0.5) * 2
    return max(0, blink_score)

def analyze_confidence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) == 0:
            return 0.0
        pitch_std = np.std(pitch_values)
        duration = librosa.get_duration(y=y, sr=sr)
        words_per_sec = 0  # placeholder (not adding extra logic as requested)
        pitch_score = min(pitch_std / 100, 1.0)
        speech_rate_score = min(words_per_sec / 4, 1.0)
        combined_confidence = 0.7 * pitch_score + 0.3 * speech_rate_score
        return combined_confidence
    except Exception as e:
        print("Confidence analysis error:", e)
        return 0.0
    
def transcribe_google(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)   # ✅ Free Google API
        # text = recognizer.recognize_sphinx(audio)     # Offine Engine
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Google API Error: {e}"

def answer_quality_score(candidate_answer, expected_answer):
    if not candidate_answer or not expected_answer:
        return 0.0
    emb1 = sbert_model.encode(candidate_answer, convert_to_tensor=True)
    emb2 = sbert_model.encode(expected_answer, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(emb1, emb2).item()
    return sim_score

def normalized_sentiment_score(text):
    if not text:
        return 0.5
    raw_score = TextBlob(text).sentiment.polarity
    return (raw_score + 1) / 2

def intelligent_response_speed_score(response_duration, expected_answer_length, max_duration=60):
    base_time = max(10, min(expected_answer_length / 2, max_duration))
    score = max(0, 1 - response_duration / base_time)
    return min(score, 1)