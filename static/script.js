// ===== Elements =====
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
let answerStartTime = 0;

// ===== Test Lock Vars =====
let testRunning = false;

// ===== Utils =====
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


// ===== Fullscreen =====
function requestFullScreen() {
  let elem = document.documentElement;
  if (elem.requestFullscreen) elem.requestFullscreen();
  else if (elem.mozRequestFullScreen) elem.mozRequestFullScreen();
  else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
  else if (elem.msRequestFullscreen) elem.msRequestFullscreen();
}

function exitFullScreen() {
  if (document.exitFullscreen) document.exitFullscreen();
  else if (document.mozCancelFullScreen) document.mozCancelFullScreen();
  else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
  else if (document.msExitFullscreen) document.msExitFullscreen();
}

// ===== Strict Violation Handling =====
function handleViolation(reason) {
  if (!testRunning) return;
  toast(`❌ Test stopped: ${reason}`);
  log(`❌ Test stopped: ${reason}`);
  endTest();
}

function startTestLock() {
  testRunning = true;
  requestFullScreen();

  // Detect fullscreen exit
  document.addEventListener("fullscreenchange", () => {
    if (testRunning && !document.fullscreenElement) handleViolation("Fullscreen exited");
  });

  // Detect tab switch/minimize
  document.addEventListener("visibilitychange", () => {
    if (testRunning && document.hidden) handleViolation("Tab switch or minimize detected");
  });

  // Detect window blur
  window.addEventListener("blur", () => {
    if (testRunning) handleViolation("Window lost focus");
  });
}

function endTest() {
  testRunning = false;
  document.removeEventListener("visibilitychange", handleViolation);
  window.removeEventListener("blur", handleViolation);
  exitFullScreen();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
}

// ===== Camera & Recording =====
async function setupCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  videoEl.srcObject = stream;
  await videoEl.play();
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;
}

// ===== Frame Loop =====
function startFrameLoop() {
  setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!videoEl.videoWidth) return;

    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;

    // Draw video frame
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

    // Send to backend for processing
    const dataUrl = canvasEl.toDataURL("image/jpeg");
    ws.send(JSON.stringify({ type: "frame", data: dataUrl }));
  }, 200);
}

// ===== Recording =====
function startRecording() {
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8,opus" });
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.start(500);
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
      form.append("expected_answer", "");
      const responseDuration = (performance.now() - answerStartTime) / 1000;
      form.append("response_duration", String(responseDuration.toFixed(2)));

      log(`Uploading answer (${(blob.size / 1024).toFixed(1)} KB)...`);
      const res = await fetch("/upload", { method: "POST", body: form });
      const data = await res.json();
      log("Upload & evaluation done");

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
  await stopRecordingAndUpload();
  ws.send(JSON.stringify({ type: "next_question" }));
}

// ===== WebSocket & Event Handling =====
startBtn.onclick = async () => {
  startBtn.disabled = true;
  startTestLock();
  await setupCamera();

  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");

  ws.onopen = () => {
    log("WebSocket connected");
    startFrameLoop();
    nextBtn.disabled = false;
    stopBtn.disabled = false;
    ws.send(JSON.stringify({ type: "next_question" }));
  };

  ws.onmessage = async (event) => {
    let payload = {};
    try { payload = JSON.parse(event.data); } catch { return; }

    // ===== Session =====
    if (payload.type === "session") {
      sessionId = payload.session_id;
      sessionEl.textContent = sessionId;
      pdfBtn.disabled = false;
    }

    // ===== Warnings =====
    else if (payload.type === "warning") {
      toast(payload.message || "Warning");
      log(`Warning: ${payload.message}`);
    }

    // ===== Questions =====
    else if (payload.type === "question") {
      currentQuestion = payload.text || "";
      questionEl.textContent = currentQuestion || "-";
      log(`Question: ${currentQuestion}`);
      if (payload.audio) {
        const audio = new Audio(payload.audio);
        audio.play().catch(() => {});
      }
      startRecording();
    }

    // ===== Stop =====
    else if (payload.type === "stop") {
      toast(payload.message || "Stopped");
      log(`Stop: ${payload.message}`);
      nextBtn.disabled = true;
      stopBtn.disabled = true;
      await stopRecordingAndUpload();
      try { ws.close(); } catch {}
      endTest();
    }

    // ===== Errors =====
    else if (payload.type === "error") {
      toast("Error: " + payload.message);
      log("Error: " + payload.message);
    }

    // ===== Frame with face box =====
    else if (payload.type === "frame_boxed") {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        ctx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
      };
      img.src = payload.data; // data:image/jpeg;base64,...
    }
  };

  ws.onclose = () => { log("WebSocket disconnected"); };
};

nextBtn.onclick = async () => {
  await handleQuestionAdvance();
};

stopBtn.onclick = async () => {
  endTest();
};

pdfBtn.onclick = async () => {
  if (!sessionId) return;
  const url = `/export_pdf?session_id=${encodeURIComponent(sessionId)}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = `interview_${sessionId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
};
