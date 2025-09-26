from datetime import datetime
import os
import shutil
import subprocess
from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import FileResponse

from .analysis import (
    analyze_confidence,
    analyze_eye_contact,
    analyze_posture,
    analyze_smile,
    detect_face_landmarks,
    extract_frames,
    calculate_blink_rate,
    normalized_sentiment_score,
    intelligent_response_speed_score
)
from .answer_evaluation import answer_quality_score
from .transcript import transcribe_google
from .database import SessionLocal, Transcript, InterviewResult
from .final_score import calculate_and_save_final_score

router = APIRouter()


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(...),
    expected_answer: str = Form(default=""),
    response_duration: float = Form(default=60.0),
    is_last_question: bool = Form(False)  # <-- Add this to detect last question
):
    # -------------------------------
    # Save video and extract audio
    # -------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp}.webm"
    video_path = os.path.join("videos", video_filename)
    audio_filename = f"{timestamp}.wav"
    audio_path = os.path.join("audio", audio_filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"FFmpeg audio extract failed: {e}"}

    # -------------------------------
    # Transcription
    # -------------------------------
    try:
        transcript_text = transcribe_google(audio_path)
    except Exception as e:
        transcript_text = f"Transcription failed: {e}"

    # -------------------------------
    # Video analysis
    # -------------------------------
    frames = extract_frames(video_path)
    landmarks_list = detect_face_landmarks(frames)
    frame_width = frames[0].shape[1] if frames else 640
    frame_height = frames[0].shape[0] if frames else 480

    eye_contact_score = analyze_eye_contact(landmarks_list, frame_width, frame_height)
    posture_score = analyze_posture(landmarks_list)
    smile_score = analyze_smile(landmarks_list)
    blink_rate_score = calculate_blink_rate(landmarks_list)
    confidence_score = analyze_confidence(audio_path)
    answer_quality = answer_quality_score(transcript_text, expected_answer)
    sentiment_score = normalized_sentiment_score(transcript_text)
    response_speed_score = intelligent_response_speed_score(response_duration, len(expected_answer.split()))

    final_engagement = (
        0.3 * answer_quality +
        0.2 * sentiment_score +
        0.2 * response_speed_score +
        0.1 * eye_contact_score +
        0.1 * posture_score +
        0.05 * smile_score +
        0.05 * blink_rate_score
    )
    final_engagement = max(0, min(final_engagement, 1))

    # -------------------------------
    # Save to database (Transcript)
    # -------------------------------
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
        answer_quality_score=answer_quality,
        sentiment_score=sentiment_score,
        response_speed_score=response_speed_score,
        final_engagement_score=final_engagement,
        smile_score=smile_score,
        blink_rate_score=blink_rate_score
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    # -------------------------------
    # Auto calculate final score if last question
    # -------------------------------
    final_result = None
    if is_last_question:
        final_result = calculate_and_save_final_score(session_id, db)

    db.close()

    # -------------------------------
    # Response
    # -------------------------------
    response = {
        "status": "success",
        "message": "Uploaded and evaluated",
        "id": new_entry.id,
        "eye_contact_score": round(eye_contact_score, 2),
        "posture_score": round(posture_score, 2),
        "confidence_score": round(confidence_score, 2),
        "answer_quality_score": round(answer_quality, 2),
        "sentiment_score": round(sentiment_score, 2),
        "response_speed_score": round(response_speed_score, 2),
        "smile_score": round(smile_score, 2),
        "blink_rate_score": round(blink_rate_score, 2),
        "final_engagement_score": round(final_engagement, 2),
        "transcript": transcript_text
    }

    if final_result:
        response["final_score"] = round(final_result.final_score, 2)

    return response
