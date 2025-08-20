import cv2
import math
import librosa
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
# Use one global instance
face_mesh_analysis = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Utility ----------
def euclidean_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ---------- Face Landmarks ----------
def detect_face_landmarks(frames, resize_dim=(320, 240)):
    landmarks_list = []
    for frame in frames:
        small_frame = cv2.resize(frame, resize_dim)  # speedup
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_analysis.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks_list.append(results.multi_face_landmarks[0])
        else:
            landmarks_list.append(None)
    return landmarks_list

# ---------- Eye Aspect Ratio ----------
def calculate_eye_aspect_ratio(landmarks, img_width, img_height):
    left_eye_indices = [33, 133, 159, 145, 153, 154]
    right_eye_indices = [362, 263, 386, 374, 380, 381]

    def eye_ratio(eye_pts):
        A = euclidean_dist((eye_pts[1].x*img_width, eye_pts[1].y*img_height),
                           (eye_pts[5].x*img_width, eye_pts[5].y*img_height))
        B = euclidean_dist((eye_pts[2].x*img_width, eye_pts[2].y*img_height),
                           (eye_pts[4].x*img_width, eye_pts[4].y*img_height))
        C = euclidean_dist((eye_pts[0].x*img_width, eye_pts[0].y*img_height),
                           (eye_pts[3].x*img_width, eye_pts[3].y*img_height))
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
    baseline = np.median(ear_values)
    adaptive_threshold = baseline * 0.8
    score = min(max((avg_ear - adaptive_threshold) * 5, 0), 1)
    return score

# ---------- Posture ----------
def analyze_posture(landmarks_list):
    tilt_angles = []
    for landmarks in landmarks_list:
        if landmarks:
            nose = landmarks.landmark[1]
            chin = landmarks.landmark[152]
            left_ear = landmarks.landmark[234]
            right_ear = landmarks.landmark[454]

            dx = chin.x - nose.x
            dy = chin.y - nose.y
            angle = math.degrees(math.atan2(dy, dx))

            ear_diff = abs(left_ear.y - right_ear.y)  # left vs right tilt
            tilt_score = 1 - min(ear_diff * 5, 1)  # normalize

            tilt_angles.append((angle, tilt_score))

    if not tilt_angles:
        return 0.0

    avg_angle = sum(a for a, _ in tilt_angles) / len(tilt_angles)
    avg_tilt_score = sum(s for _, s in tilt_angles) / len(tilt_angles)

    forward_score = max(0, 1 - abs(avg_angle - 90) / 45)
    final_score = (0.6 * forward_score) + (0.4 * avg_tilt_score)
    return final_score

# ---------- Smile ----------
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
                smile_score = max(0, min(1, 0.4 - mar))
                scores.append(smile_score)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)

# ---------- Blink Rate ----------
def calculate_blink_rate(landmarks_list, fps=30, frame_skip=2):
    blink_frames = 0
    total_frames = len(landmarks_list)
    EAR_THRESHOLD = 0.21

    for i, landmarks in enumerate(landmarks_list):
        if i % frame_skip != 0:
            continue
        if landmarks:
            img_w, img_h = 320, 240
            ear = calculate_eye_aspect_ratio(landmarks, img_w, img_h)
            if ear < EAR_THRESHOLD:
                blink_frames += 1

    effective_frames = total_frames // frame_skip
    blink_rate_per_sec = (blink_frames / effective_frames) * fps if effective_frames > 0 else 0
    normalized_blink = min(max((blink_rate_per_sec - 0.25) / 0.25, 0), 1)
    blink_score = 1 - abs(normalized_blink - 0.5) * 2
    return max(0, blink_score)

# ---------- Audio Confidence ----------
def analyze_confidence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)  # faster pitch extraction
        pitch_std = np.std(f0)
        duration = librosa.get_duration(y=y, sr=sr)

        pitch_score = min(pitch_std / 80, 1.0)

        # simple speech rate estimate by energy bursts
        energy = librosa.feature.rms(y=y)[0]
        speech_segments = np.sum(energy > np.median(energy))
        words_per_sec = speech_segments / duration if duration > 0 else 0
        speech_rate_score = min(words_per_sec / 4, 1.0)

        combined_confidence = 0.7 * pitch_score + 0.3 * speech_rate_score
        return combined_confidence
    except Exception as e:
        print("Confidence analysis error:", e)
        return 0.0
