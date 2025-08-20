# questions_utils.py
import os, time, json
import edge_tts

# Questions list
questions = [
    "Q1: What is your name?",
    "Q2: Where do you live?",
    "Q3: What is your favorite programming language?",
    "Q4: Tell me about your hobbies.",
    "Q5: What is your goal for this year?"
]

# Session tracking
extra_person_count: dict[str, int] = {}
sessions_state: dict[str, dict] = {}

# TTS function
async def run_tts(question_text, session_id):
    filename = f"tts_{session_id}_{int(time.time())}.mp3"
    filepath = os.path.join("audio", filename)
    communicate = edge_tts.Communicate(question_text, "en-US-AriaNeural")
    await communicate.save(filepath)
    return f"/audio/{filename}"

# Next question helper
async def get_next_question(session_id: str):
    idx = sessions_state[session_id]["question_index"]
    if idx < len(questions):
        question = questions[idx]
        sessions_state[session_id]["question_index"] += 1
        sessions_state[session_id]["active_question"] = question
        audio_url = await run_tts(question, session_id)
        return {
            "type": "question",
            "text": question,
            "audio": audio_url,
            "session_id": session_id,
            "index": idx
        }
    else:
        return {"type": "stop", "message": "Interview completed"}
