from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from database import SessionLocal
from models import Transcript
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
import os

router = APIRouter()

@router.get("/export_pdf")
async def export_pdf(session_id: str = Query(...)):
    db = SessionLocal()
    rows = db.query(Transcript).filter(Transcript.session_id == session_id).order_by(Transcript.id.asc()).all()
    db.close()

    if not rows:
        return {"status": "error", "message": "No data for this session"}

    os.makedirs("pdf", exist_ok=True)
    file_path = os.path.join("pdf", f"interview_{session_id}.pdf")
    c = pdf_canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Interview Report - Session: {session_id}")
    y -= 30

    c.setFont("Helvetica", 11)
    for i, row in enumerate(rows, start=1):
        lines = [
            f"{i}. {row.question}",
            f"Answer: {row.transcript}",
            f"Eye Contact: {row.eye_contact_score:.2f} | Posture: {row.posture_score:.2f} | Confidence: {row.confidence_score:.2f}",
            f"Answer Quality: {row.answer_quality_score:.2f} | Sentiment: {row.sentiment_score:.2f} | Speed: {row.response_speed_score:.2f}",
            f"Smile: {row.smile_score:.2f} | Blink: {row.blink_rate_score:.2f} | Final Engagement: {row.final_engagement_score:.2f}",
        ]
        for ln in lines:
            c.drawString(50, y, ln[:110])
            y -= 16
            if y < 60:
                c.showPage(); y = height - 50; c.setFont("Helvetica", 11)
        y -= 8
        if y < 60:
            c.showPage(); y = height - 50; c.setFont("Helvetica", 11)

    c.save()
    return FileResponse(file_path, filename=f"interview_{session_id}.pdf", media_type="application/pdf")
