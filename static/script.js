const videoEl = document.getElementById("preview");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const nextBtn = document.getElementById("nextQuestionBtn");
const pdfBtn = document.getElementById("downloadPdfBtn");

const warningDiv = document.getElementById("warningMsg");
const statusLog = document.getElementById("statusLog");
const questionEl = document.getElementById("question");
const sessionEl = document.getElementById("sessionId");
const transcriptsDiv = document.getElementById("transcripts");

let ws;
let stream;
let mediaRecorder;
let recordedChunks = [];
let sessionId = "";
let currentQuestion = "";
let answerStartTime = 0; // for response_duration

function log(msg) {
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  statusLog.prepend(line);
}

function toast(msg) {
  warningDiv.textContent = msg;
  warningDiv.style.opacity = 1;
  setTimeout(() => (warningDiv.style.opacity = 0), 1600);
}

async function setupCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  videoEl.srcObject = stream;
  await videoEl.play();
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;
}

function startFrameLoop() {
  setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!videoEl.videoWidth) return;
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    const dataUrl = canvasEl.toDataURL("image/jpeg");
    ws.send(JSON.stringify({ type: "frame", data: dataUrl }));
  }, 200);
}

function startRecording() {
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8,opus" });
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.start(500); // gather chunks
  answerStartTime = performance.now();
}

function stopRecordingAndUpload() {
  return new Promise((resolve) => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") return resolve(null);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const form = new FormData();
      form.append("file", blob, `answer_${Date.now()}.webm`);
      form.append("question", currentQuestion || "");
      form.append("session_id", sessionId);
      form.append("expected_answer", ""); // keeping empty as requested
      const responseDuration = (performance.now() - answerStartTime) / 1000;
      form.append("response_duration", String(responseDuration.toFixed(2)));

      log(`Uploading answer (${(blob.size/1024).toFixed(1)} KB)...`);
      const res = await fetch("/upload", { method: "POST", body: form });
      const data = await res.json();
      log("Upload & evaluation done");

      // show transcript + scores
      const card = document.createElement("div");
      card.style.border = "1px solid #e5e7eb";
      card.style.borderRadius = "8px";
      card.style.padding = "10px";
      card.style.marginBottom = "8px";
      card.innerHTML = `
        <div><strong>Q:</strong> ${currentQuestion || "-"}</div>
        <div><strong>Transcript:</strong> ${data.transcript || "-"}</div>
        <div style="font-size:13px; margin-top:6px;">
          Eye: ${data.eye_contact_score} | Posture: ${data.posture_score} |
          Conf: ${data.confidence_score} | AnsQual: ${data.answer_quality_score} |
          Sent: ${data.sentiment_score} | Speed: ${data.response_speed_score} |
          Smile: ${data.smile_score} | Blink: ${data.blink_rate_score} |
          Final: <strong>${data.final_engagement_score}</strong>
        </div>
      `;
      transcriptsDiv.prepend(card);

      resolve(data);
    };
    mediaRecorder.stop();
  });
}

async function handleQuestionAdvance() {
  // stop & upload the previous segment (if any)
  await stopRecordingAndUpload();
  // ask backend for next question
  ws.send(JSON.stringify({ type: "next_question" }));
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  await setupCamera();

  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => {
    log("WebSocket connected");
    startFrameLoop();
    nextBtn.disabled = false;
    stopBtn.disabled = false;
    // auto-ask first question
    ws.send(JSON.stringify({ type: "next_question" }));
  };

  ws.onmessage = async (event) => {
    let payload = {};
    try { payload = JSON.parse(event.data); } catch { return; }

    if (payload.type === "session") {
      sessionId = payload.session_id;
      sessionEl.textContent = sessionId;
      pdfBtn.disabled = false;
    }
    else if (payload.type === "warning") {
      toast(payload.message || "Warning");
      log(`Warning: ${payload.message}`);
    }
    else if (payload.type === "question") {
      currentQuestion = payload.text || "";
      questionEl.textContent = currentQuestion || "-";
      log(`Question: ${currentQuestion}`);
      // play TTS
      if (payload.audio) {
        const audio = new Audio(payload.audio);
        audio.play().catch(() => {});
      }
      // start a fresh recording segment for this question
      startRecording();
    }
    else if (payload.type === "stop") {
      toast(payload.message || "Stopped");
      log(`Stop: ${payload.message}`);
      nextBtn.disabled = true;
      stopBtn.disabled = true;
      await stopRecordingAndUpload(); // final upload
      try { ws.close(); } catch {}
    }
    else if (payload.type === "error") {
      toast("Error: " + payload.message);
      log("Error: " + payload.message);
    }
  };

  ws.onclose = () => { log("WebSocket disconnected"); };
};

nextBtn.onclick = async () => {
  await handleQuestionAdvance();
};

stopBtn.onclick = async () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
  }
};

pdfBtn.onclick = async () => {
  if (!sessionId) return;
  const url = `/export_pdf?session_id=${encodeURIComponent(sessionId)}`;
  // force download
  const a = document.createElement("a");
  a.href = url;
  a.download = `interview_${sessionId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
};
