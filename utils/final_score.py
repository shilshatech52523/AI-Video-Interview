from .database import Transcript, InterviewResult

def calculate_and_save_final_score(session_id: str, db):
    """
    Calculates the final weighted score for all transcripts in a session,
    converts it to percentage (0–100), and saves/updates it in InterviewResult table.
    """

    # 1️⃣ Fetch all transcripts for the session
    transcripts = db.query(Transcript).filter(Transcript.session_id == session_id).all()
    if not transcripts:
        return None

    # 2️⃣ Weighted score calculation
    total_score = 0
    count = 0
    for t in transcripts:
        score = (
            (t.eye_contact_score * 0.1) +
            (t.posture_score * 0.1) +
            (t.confidence_score * 0.2) +
            (t.answer_quality_score * 0.3) +
            (t.sentiment_score * 0.1) +
            (t.response_speed_score * 0.1) +
            (t.final_engagement_score * 0.05) +
            (t.smile_score * 0.025) +
            (t.blink_rate_score * 0.025)
        )
        total_score += score
        count += 1

    # 3️⃣ Convert final score to percentage
    final_score = (total_score / count if count > 0 else 0) * 100
    final_score = max(0, min(final_score, 100))  # Clamp between 0–100

    # 4️⃣ Check if result already exists in DB
    existing_result = db.query(InterviewResult).filter(InterviewResult.session_id == session_id).first()
    if existing_result:
        existing_result.final_score = final_score
        db.commit()
        db.refresh(existing_result)
        return existing_result

    # 5️⃣ Create new InterviewResult row
    result = InterviewResult(session_id=session_id, final_score=final_score)
    db.add(result)
    db.commit()
    db.refresh(result)
    return result
