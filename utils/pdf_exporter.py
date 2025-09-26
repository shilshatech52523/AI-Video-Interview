import os
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from .database import SessionLocal, Transcript, InterviewResult

router = APIRouter()

@router.get("/export_pdf")
async def export_pdf(session_id: str = Query(...)):
    db = SessionLocal()
    try:
        # Fetch per-question transcripts
        rows = db.query(Transcript).filter(Transcript.session_id == session_id).order_by(Transcript.id.asc()).all()
        # Fetch final overall score
        final_result = db.query(InterviewResult).filter(InterviewResult.session_id == session_id).first()
    finally:
        db.close()

    if not rows:
        return {"status": "error", "message": "No data for this session"}

    # Ensure pdf directory exists
    os.makedirs("pdf", exist_ok=True)

    # PDF file path
    file_path = os.path.abspath(os.path.join("pdf", f"interview_{session_id}.pdf"))

    # Create PDF
    c = pdf_canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Interview Report - Session: {session_id}")
    y -= 25

    # Add final score and pass/fail status at top
    if final_result:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Final Score: {final_result.final_score:.2f}")
        y -= 20
        pass_status = "Passed ✅" if final_result.passed == "Pass" else "Failed ❌"
        c.drawString(50, y, f"Status: {pass_status}")
        y -= 25

    # Set font for details
    c.setFont("Helvetica", 11)

    # Add per-question details
    for i, row in enumerate(rows, start=1):
        lines = [
            f"{i}. {row.question}",
            f"Answer: {row.transcript}",
            f"Eye Contact: {row.eye_contact_score:.2f} | Posture: {row.posture_score:.2f} | Confidence: {row.confidence_score:.2f}",
            f"Answer Quality: {row.answer_quality_score:.2f} | Sentiment: {row.sentiment_score:.2f} | Speed: {row.response_speed_score:.2f}",
            f"Smile: {row.smile_score:.2f} | Blink: {row.blink_rate_score:.2f} | Final Engagement: {row.final_engagement_score:.2f}",
        ]
        for ln in lines:
            c.drawString(50, y, ln[:110])  # Limit line length
            y -= 16
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
        y -= 8
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)

    # Save PDF
    try:
        c.save()
    except Exception as e:
        return {"status": "error", "message": f"Failed to save PDF: {str(e)}"}

    if not os.path.isfile(file_path):
        return {"status": "error", "message": "PDF file was not created."}

    return FileResponse(file_path, filename=f"interview_{session_id}.pdf", media_type="application/pdf")
