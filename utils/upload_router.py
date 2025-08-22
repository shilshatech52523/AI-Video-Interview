from .analysis import analyze_confidence,analyze_eye_contact,analyze_posture,analyze_smile,answer_quality_score,face_mesh_analysis,calculate_blink_rate,SentenceTransformer,normalized_sentiment_score,intelligent_response_speed_score,detect_face_landmarks,extract_frames,transcribe_google
from .database import sessionmaker,SessionLocal,Transcript
from fastapi import APIRouter, Query,Form,File,UploadFile
from fastapi.responses import FileResponse
from datetime import datetime
import os, shutil, subprocess, json
import datetime as dt
from vosk import Model, KaldiRecognizer

router = APIRouter()

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(...),
    expected_answer: str = Form(default=""),
    response_duration: float = Form(default=60.0)
):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".webm"
    video_path = os.path.join("videos", filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio_filename = filename.replace(".webm", ".wav")
    audio_path = os.path.join("audio", audio_filename)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"FFmpeg audio extract failed: {e}"}


    try:
        transcript_text = transcribe_google(audio_path)
    except Exception as e:
        transcript_text = f"Transcription failed: {e}"


    frames = extract_frames(video_path)
    landmarks_list = detect_face_landmarks(frames)

    frame_width = frames[0].shape[1] if frames else 640
    frame_height = frames[0].shape[0] if frames else 480

    eye_contact_score = analyze_eye_contact(landmarks_list, frame_width, frame_height)
    posture_score = analyze_posture(landmarks_list)
    smile_score = analyze_smile(landmarks_list)
    blink_rate_score = calculate_blink_rate(landmarks_list)
    confidence_score = analyze_confidence(audio_path)
    a_quality_score = answer_quality_score(transcript_text, expected_answer)
    sent_score = normalized_sentiment_score(transcript_text)
    resp_speed_score = intelligent_response_speed_score(response_duration, len(expected_answer.split()))

    final_engagement = (
        0.3 * a_quality_score +
        0.2 * sent_score +
        0.2 * resp_speed_score +
        0.1 * eye_contact_score +
        0.1 * posture_score +
        0.05 * smile_score +
        0.05 * blink_rate_score
    )
    final_engagement = min(max(final_engagement, 0), 1)

    db = SessionLocal()
    new_entry = Transcript(
        session_id=session_id,
        question=question,
        video_path=video_path,
        audio_path=audio_path,
        transcript=transcript_text,
        eye_contact_score=eye_contact_score,
        posture_score=posture_score,
        confidence_score=confidence_score,
        answer_quality_score=a_quality_score,
        sentiment_score=sent_score,
        response_speed_score=resp_speed_score,
        final_engagement_score=final_engagement,
        smile_score=smile_score,
        blink_rate_score=blink_rate_score
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    db.close()

    return {
        "status": "success",
        "message": "Uploaded and evaluated",
        "id": new_entry.id,
        "eye_contact_score": round(eye_contact_score, 2),
        "posture_score": round(posture_score, 2),
        "confidence_score": round(confidence_score, 2),
        "answer_quality_score": round(a_quality_score, 2),
        "sentiment_score": round(sent_score, 2),
        "response_speed_score": round(resp_speed_score, 2),
        "smile_score": round(smile_score, 2),
        "blink_rate_score": round(blink_rate_score, 2),
        "final_engagement_score": round(final_engagement, 2),
        "transcript": transcript_text
    }



