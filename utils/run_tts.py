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





# # tts.py
# import os
# import time
# from pathlib import Path
# import requests

# # Replace with your actual ElevenLabs API key and preferred voice
# ELEVENLABS_API_KEY = "YOUR_ELEVENLABS_API_KEY"
# ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # default US English voice

# def ensure_audio_folder():
#     Path("audio").mkdir(exist_ok=True)

# def run_tts(question_text: str, session_id: str) -> str:
#     """
#     Generates speech from text using ElevenLabs natural voice.
#     Returns the local URL to the generated MP3 file.
#     """
#     ensure_audio_folder()
    
#     # Unique filename
#     filename = f"tts_{session_id}_{int(time.time())}.mp3"
#     filepath = os.path.join("audio", filename)
    
#     # ElevenLabs TTS API endpoint
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
#     headers = {
#         "xi-api-key": ELEVENLABS_API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "text": question_text,
#         "voice_settings": {
#             "stability": 0.75,   # 0 to 1
#             "similarity_boost": 0.75
#         }
#     }

#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         # Save MP3
#         with open(filepath, "wb") as f:
#             f.write(response.content)
#         return f"/audio/{filename}"
#     else:
#         raise Exception(f"ElevenLabs TTS failed: {response.status_code}, {response.text}")
