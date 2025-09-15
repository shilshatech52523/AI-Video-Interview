
import speech_recognition as sr

def transcribe_google(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)   
        # text = recognizer.recognize_sphinx(audio)     # Offine Engine
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Google API Error: {e}"