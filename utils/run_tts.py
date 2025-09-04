import pyttsx3
import time
import os


def run_tts(question_text, session_id):
    # Use WAV for maximum reliability with pyttsx3
    filename = f"tts_{session_id}_{int(time.time())}.wav"
    filepath = os.path.join("audio", filename)
    engine = pyttsx3.init()
    engine.save_to_file(question_text, filepath)
    engine.runAndWait()
    # Return URL the frontend can fetch
    return f"/audio/{filename}"
