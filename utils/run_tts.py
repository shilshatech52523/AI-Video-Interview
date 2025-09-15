import edge_tts
import time
import os

async def run_tts(question_text, session_id, voice="en-US-AriaNeural"):
    # Ensure 'audio' folder exists
    if not os.path.exists("audio"):
        os.makedirs("audio")

    # Unique file name
    filename = f"tts_{session_id}_{int(time.time())}.mp3"
    filepath = os.path.join("audio", filename)

    # Generate speech with Microsoft Neural voice
    communicate = edge_tts.Communicate(text=question_text, voice=voice)
    await communicate.save(filepath)

    # Return URL for frontend
    return f"/audio/{filename}"
