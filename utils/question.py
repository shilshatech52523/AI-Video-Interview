# question.py
import requests, html

def get_questions():
    url = "https://opentdb.com/api.php?amount=1"
    response = requests.get(url)
    data = response.json()

    if data["response_code"] == 0:
        question = html.unescape(data["results"][0]["question"])
        correct_answer = html.unescape(data["results"][0]["correct_answer"])
        return question, correct_answer   # ✅ exactly 2 return
    else:
        return "No question found.", ""
