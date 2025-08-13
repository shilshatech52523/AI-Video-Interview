// const videoElement = document.getElementById("preview");
// const canvas = document.getElementById("overlay");
// const ctx = canvas.getContext("2d");

// const startBtn = document.getElementById("startBtn");
// const stopBtn = document.getElementById("stopBtn");
// const nextQuestionBtn = document.getElementById("nextQuestionBtn");

// let mediaRecorder;
// let recordedChunks = [];
// let questionIndex = 0;
// let currentQuestion = "";
// let sessionId = "";
// let randomScreenshotInterval = null;

// const warningDiv = document.getElementById("warningMsg");
// const ws = new WebSocket("ws://localhost:8000/cheating_status");

// ws.onmessage = (event) => {
//     const data = JSON.parse(event.data);
//     if (data.cheating) {
//         stopRecording(); // Your own stop function
//         alert("Cheating detected! Recording stopped.");
//         ws.close();  // Optional
//     }
// };

// function logStatus(message) {
//     const logDiv = document.getElementById("statusLog");
//     const time = new Date().toLocaleTimeString();
//     logDiv.innerHTML += `<div>[${time}] ${message}</div>`;
// }


// function showWarning(message) {
//     warningDiv.textContent = message;
//     warningDiv.style.opacity = '1';
//     setTimeout(() => {
//         warningDiv.style.transition = "opacity 1s";
//         warningDiv.style.opacity = '0';
//     }, 4000);
// }

// const questions = [
//     "Q1: What is your name?",
//     "Q2: Where do you live?",
//     "Q3: What is your favorite programming language?",
//     "Q4: Tell me about your hobbies.",
//     "Q5: What is your goal for this year?"
// ];

// const faceMesh = new FaceMesh({
//     locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
// });
// faceMesh.setOptions({
//     maxNumFaces: 1,
//     refineLandmarks: true,
//     minDetectionConfidence: 0.7,
//     minTrackingConfidence: 0.7
// });
// faceMesh.onResults(onResults);

// const BLINK_EAR_THRESHOLD = 0.22;
// const NO_BLINK_MAX_FRAMES = 150;
// const NO_MOVEMENT_MAX_FRAMES = 150;

// let previousLandmarks = null;
// let noBlinkFrames = 0;
// let noMovementFrames = 0;
// let blinkDetected = false;
// let facePresent = false;

// function calculateEAR(eye) {
//     function euclideanDist(p1, p2) {
//         return Math.hypot(p1.x - p2.x, p1.y - p2.y);
//     }
//     const A = euclideanDist(eye[1], eye[5]);
//     const B = euclideanDist(eye[2], eye[4]);
//     const C = euclideanDist(eye[0], eye[3]);
//     return (A + B) / (2.0 * C);
// }

// function getEyeLandmarks(landmarks, eyeIndices) {
//     return eyeIndices.map(i => landmarks[i]);
// }

// function drawFaceBoundingBox(landmarks) {
//     if (!landmarks) {
//         ctx.clearRect(0, 0, canvas.width, canvas.height);
//         return;
//     }
//     const xs = landmarks.map(lm => lm.x * canvas.width);
//     const ys = landmarks.map(lm => lm.y * canvas.height);

//     const minX = Math.min(...xs);
//     const maxX = Math.max(...xs);
//     const minY = Math.min(...ys);
//     const maxY = Math.max(...ys);

//     ctx.clearRect(0, 0, canvas.width, canvas.height);
//     ctx.strokeStyle = '#00FF00';
//     ctx.lineWidth = 3;
//     ctx.beginPath();
//     ctx.rect(minX, minY, maxX - minX, maxY - minY);
//     ctx.stroke();
// }

// function onResults(results) {
//     const faces = results.multiFaceLandmarks || [];

//     if (faces.length === 0) {
//         facePresent = false;
//         showWarning("No face detected. Please show your face clearly.");
//         ctx.clearRect(0, 0, canvas.width, canvas.height);
//         resetLiveness();
//         return;
//     } else if (faces.length > 1) {
//         facePresent = true;
//         showWarning("Multiple faces detected! Please ensure only one person is in front of the camera.");
//         ctx.clearRect(0, 0, canvas.width, canvas.height);
//         faces.forEach(landmarks => {
//             if (!landmarks) return;
//             const xs = landmarks.map(lm => lm.x * canvas.width);
//             const ys = landmarks.map(lm => lm.y * canvas.height);
//             const minX = Math.min(...xs);
//             const maxX = Math.max(...xs);
//             const minY = Math.min(...ys);
//             const maxY = Math.max(...ys);

//             ctx.strokeStyle = '#FF0000';
//             ctx.lineWidth = 3;
//             ctx.beginPath();
//             ctx.rect(minX, minY, maxX - minX, maxY - minY);
//             ctx.stroke();
//         });
//         resetLiveness();
//         return;
//     }

//     facePresent = true;
//     const landmarks = faces[0];
//     drawFaceBoundingBox(landmarks);

//     const leftEyeIndices = [33, 160, 158, 133, 153, 144];
//     const rightEyeIndices = [362, 385, 387, 263, 373, 380];

//     const leftEye = getEyeLandmarks(landmarks, leftEyeIndices);
//     const rightEye = getEyeLandmarks(landmarks, rightEyeIndices);

//     const leftEAR = calculateEAR(leftEye);
//     const rightEAR = calculateEAR(rightEye);
//     const avgEAR = (leftEAR + rightEAR) / 2.0;

//     if (!blinkDetected && avgEAR < BLINK_EAR_THRESHOLD) {
//         blinkDetected = true;
//         noBlinkFrames = 0;
//     } else if (!blinkDetected) {
//         noBlinkFrames++;
//     } else if (avgEAR > BLINK_EAR_THRESHOLD) {
//         blinkDetected = false;
//     }

//     if (noBlinkFrames > NO_BLINK_MAX_FRAMES) {
//         showWarning("Please blink your eyes to verify liveness!");
//         noBlinkFrames = 0;
//     }

//     let movementSum = 0;
//     if (previousLandmarks) {
//         for (let i = 0; i < landmarks.length; i++) {
//             const dx = landmarks[i].x - previousLandmarks[i].x;
//             const dy = landmarks[i].y - previousLandmarks[i].y;
//             movementSum += Math.sqrt(dx * dx + dy * dy);
//         }
//     } else {
//         movementSum = 1;
//     }

//     if (movementSum > 0.001) {
//         noMovementFrames = 0;
//     } else {
//         noMovementFrames++;
//     }

//     if (noMovementFrames > NO_MOVEMENT_MAX_FRAMES) {
//         showWarning("Please move your head slightly to verify liveness!");
//         noMovementFrames = 0;
//     }

//     previousLandmarks = landmarks;
// }

// function resetLiveness() {
//     previousLandmarks = null;
//     noBlinkFrames = 0;
//     noMovementFrames = 0;
//     blinkDetected = false;
// }

// async function initCamera() {
//     const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
//     videoElement.srcObject = stream;

//     return new Promise((resolve) => {
//         videoElement.onloadedmetadata = () => {
//             canvas.width = videoElement.videoWidth;
//             canvas.height = videoElement.videoHeight;

//             const camera = new Camera(videoElement, {
//                 onFrame: async () => {
//                     await faceMesh.send({ image: videoElement });
//                 },
//                 width: videoElement.videoWidth,
//                 height: videoElement.videoHeight,
//             });
//             camera.start();
//             resolve(stream);
//         };
//     });
// }

// function speak(text) {
//     const utterance = new SpeechSynthesisUtterance(text);
//     utterance.lang = "en-US";
//     utterance.rate = 1;
//     utterance.pitch = 1;
//     window.speechSynthesis.speak(utterance);
// }

// function captureScreenshot() {
//     const offscreenCanvas = document.createElement("canvas");
//     offscreenCanvas.width = videoElement.videoWidth;
//     offscreenCanvas.height = videoElement.videoHeight;

//     const context = offscreenCanvas.getContext("2d");
//     context.drawImage(videoElement, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

//     offscreenCanvas.toBlob(async (blob) => {
//         const formData = new FormData();
//         formData.append("file", blob, "screenshot.jpg");

//         await fetch("/save_screenshot", {
//             method: "POST",
//             body: formData
//         });

//         showWarning("📸 Screenshot taken!");
//     }, "image/jpeg", 0.95);
// }

// async function startRecording(questionText) {
//     const stream = await initCamera();
//     mediaRecorder = new MediaRecorder(stream);
//     recordedChunks = [];

//     mediaRecorder.ondataavailable = (event) => {
//         if (event.data.size > 0) recordedChunks.push(event.data);
//     };

//     mediaRecorder.onstop = async () => {
//         if (recordedChunks.length > 0) {
//             const blob = new Blob(recordedChunks, { type: "video/webm" });
//             const formData = new FormData();
//             formData.append("file", blob, "recording.webm");
//             formData.append("question", questionText);
//             formData.append("session_id", sessionId);

//             const response = await fetch("/upload", {
//                 method: "POST",
//                 body: formData
//             });
//             console.log("Saved:", await response.json());
//         }
//     };

//     mediaRecorder.start();
//     speak(questionText);
// }

// function stopCurrentRecording() {
//     if (mediaRecorder && mediaRecorder.state !== "inactive") {
//         mediaRecorder.stop();
//     }
// }

// startBtn.onclick = () => {
//     sessionId = crypto.randomUUID();
//     questionIndex = 0;
//     startBtn.disabled = true;
//     nextQuestionBtn.disabled = false;
//     stopBtn.disabled = false;

//     currentQuestion = questions[questionIndex];
//     startRecording(currentQuestion);
//     captureScreenshot(); // Take immediate screenshot

//     randomScreenshotInterval = setInterval(() => {
//         captureScreenshot();
//     }, Math.floor(Math.random() * 5000) + 5000); // Random every 5–10s

//     questionIndex++;
// };

// nextQuestionBtn.onclick = () => {
//     stopCurrentRecording();

//     if (questionIndex < questions.length) {
//         currentQuestion = questions[questionIndex];
//         startRecording(currentQuestion);
//         questionIndex++;
//     } else {
//         alert("Interview Completed! Please click Stop.");
//         nextQuestionBtn.disabled = true;
//     }
// };

// stopBtn.onclick = async () => {
//     stopCurrentRecording();
//     clearInterval(randomScreenshotInterval);
//     nextQuestionBtn.disabled = true;
//     startBtn.disabled = false;
//     await fetch('/stop_camera', { method: 'POST' });
//     document.getElementById('cameraFeed').src = '';


//     try {
//         const res = await fetch(`/export_pdf?session_id=${sessionId}`);
//         const blob = await res.blob();
//         const url = window.URL.createObjectURL(blob);

//         const a = document.createElement("a");
//         a.href = url;
//         a.download = "interview_report.pdf";
//         document.body.appendChild(a);
//         a.click();
//         a.remove();
//     } catch (err) {
//         console.error("PDF export failed", err);
//     }
// };




////////////////////////////////////////////




const videoElement = document.getElementById("preview");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const nextQuestionBtn = document.getElementById("nextQuestionBtn");

let mediaRecorder;
let recordedChunks = [];
let questionIndex = 0;
let currentQuestion = "";
let sessionId = "";
let randomScreenshotTimeout = null;

const warningDiv = document.getElementById("warningMsg");

// ✅ WebSocket for real-time cheating detection
let ws;
function initWebSocket() {
    ws = new WebSocket("ws://localhost:8000/cheating_status");
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.cheating) {
            stopCurrentRecording();
            alert("⚠ Cheating detected! Recording stopped.");
        }
    };
    ws.onclose = () => console.log("WebSocket closed");
}

function logStatus(message) {
    const logDiv = document.getElementById("statusLog");
    const time = new Date().toLocaleTimeString();
    logDiv.innerHTML += `<div>[${time}] ${message}</div>`;
}

function showWarning(message) {
    warningDiv.textContent = message;
    warningDiv.style.opacity = '1';
    setTimeout(() => {
        warningDiv.style.transition = "opacity 1s";
        warningDiv.style.opacity = '0';
    }, 4000);
}

const questions = [
    "Q1: What is your name?",
    "Q2: Where do you live?",
    "Q3: What is your favorite programming language?",
    "Q4: Tell me about your hobbies.",
    "Q5: What is your goal for this year?"
];

// ✅ Mediapipe FaceMesh setup
const faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});
faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});
faceMesh.onResults(onResults);

const BLINK_EAR_THRESHOLD = 0.22;
const NO_BLINK_MAX_FRAMES = 150;
const NO_MOVEMENT_MAX_FRAMES = 150;

let previousLandmarks = null;
let noBlinkFrames = 0;
let noMovementFrames = 0;
let blinkDetected = false;
let facePresent = false;

// EAR & landmark functions
function calculateEAR(eye) {
    function euclideanDist(p1, p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }
    const A = euclideanDist(eye[1], eye[5]);
    const B = euclideanDist(eye[2], eye[4]);
    const C = euclideanDist(eye[0], eye[3]);
    return (A + B) / (2.0 * C);
}
function getEyeLandmarks(landmarks, eyeIndices) {
    return eyeIndices.map(i => landmarks[i]);
}

// Draw face bounding box
function drawFaceBoundingBox(landmarks, color = '#00FF00') {
    if (!landmarks) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }
    const xs = landmarks.map(lm => lm.x * canvas.width);
    const ys = landmarks.map(lm => lm.y * canvas.height);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.rect(minX, minY, maxX - minX, maxY - minY);
    ctx.stroke();
}

// ✅ Face & liveness detection
function onResults(results) {
    const faces = results.multiFaceLandmarks || [];

    if (faces.length === 0) {
        facePresent = false;
        showWarning("No face detected. Please show your face clearly.");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resetLiveness();
        return;
    } else if (faces.length > 1) {
        facePresent = true;
        showWarning("Multiple faces detected! Please ensure only one person is in front of the camera.");
        faces.forEach(landmarks => drawFaceBoundingBox(landmarks, '#FF0000'));
        resetLiveness();
        return;
    }

    facePresent = true;
    const landmarks = faces[0];
    drawFaceBoundingBox(landmarks);

    const leftEyeIndices = [33, 160, 158, 133, 153, 144];
    const rightEyeIndices = [362, 385, 387, 263, 373, 380];

    const leftEye = getEyeLandmarks(landmarks, leftEyeIndices);
    const rightEye = getEyeLandmarks(landmarks, rightEyeIndices);

    const leftEAR = calculateEAR(leftEye);
    const rightEAR = calculateEAR(rightEye);
    const avgEAR = (leftEAR + rightEAR) / 2.0;

    if (!blinkDetected && avgEAR < BLINK_EAR_THRESHOLD) {
        blinkDetected = true;
        noBlinkFrames = 0;
    } else if (!blinkDetected) {
        noBlinkFrames++;
    } else if (avgEAR > BLINK_EAR_THRESHOLD) {
        blinkDetected = false;
    }

    if (noBlinkFrames > NO_BLINK_MAX_FRAMES) {
        showWarning("Please blink your eyes to verify liveness!");
        noBlinkFrames = 0;
    }

    let movementSum = 0;
    if (previousLandmarks) {
        for (let i = 0; i < landmarks.length; i++) {
            const dx = landmarks[i].x - previousLandmarks[i].x;
            const dy = landmarks[i].y - previousLandmarks[i].y;
            movementSum += Math.sqrt(dx * dx + dy * dy);
        }
    } else movementSum = 1;

    if (movementSum > 0.001) noMovementFrames = 0;
    else noMovementFrames++;

    if (noMovementFrames > NO_MOVEMENT_MAX_FRAMES) {
        showWarning("Please move your head slightly to verify liveness!");
        noMovementFrames = 0;
    }

    previousLandmarks = landmarks;
}

function resetLiveness() {
    previousLandmarks = null;
    noBlinkFrames = 0;
    noMovementFrames = 0;
    blinkDetected = false;
}

// ✅ Camera initialization
async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    videoElement.srcObject = stream;

    return new Promise((resolve) => {
        videoElement.onloadedmetadata = () => {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;

            const camera = new Camera(videoElement, {
                onFrame: async () => await faceMesh.send({ image: videoElement }),
                width: videoElement.videoWidth,
                height: videoElement.videoHeight,
            });
            camera.start();
            resolve(stream);
        };
    });
}

// ✅ TTS
function speak(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
}

// ✅ Screenshot capture
function captureScreenshot() {
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = videoElement.videoWidth;
    offscreenCanvas.height = videoElement.videoHeight;
    const context = offscreenCanvas.getContext("2d");
    context.drawImage(videoElement, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    offscreenCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "screenshot.jpg");
        await fetch("/save_screenshot", { method: "POST", body: formData });
        showWarning("📸 Screenshot taken!");
    }, "image/jpeg", 0.95);

    scheduleNextScreenshot(); // Schedule next random screenshot
}

// ✅ Random screenshots using recursive setTimeout
function scheduleNextScreenshot() {
    const interval = Math.floor(Math.random() * 5000) + 5000;
    randomScreenshotTimeout = setTimeout(captureScreenshot, interval);
}

// ✅ Recording functions
async function startRecording(questionText) {
    const stream = await initCamera();
    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
        if (recordedChunks.length > 0) {
            const blob = new Blob(recordedChunks, { type: "video/webm" });
            const formData = new FormData();
            formData.append("file", blob, "recording.webm");
            formData.append("question", questionText);
            formData.append("session_id", sessionId);

            const response = await fetch("/upload", { method: "POST", body: formData });
            console.log("Saved:", await response.json());
        }
    };

    mediaRecorder.start();
    speak(questionText);
}

function stopCurrentRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
}

// ✅ Start / Next / Stop button handlers
startBtn.onclick = () => {
    initWebSocket(); // Initialize WS for cheating
    sessionId = crypto.randomUUID();
    questionIndex = 0;
    startBtn.disabled = true;
    nextQuestionBtn.disabled = false;
    stopBtn.disabled = false;

    currentQuestion = questions[questionIndex];
    startRecording(currentQuestion);
    captureScreenshot(); // immediate first screenshot

    questionIndex++;
};

nextQuestionBtn.onclick = () => {
    stopCurrentRecording();
    if (questionIndex < questions.length) {
        currentQuestion = questions[questionIndex];
        startRecording(currentQuestion);
        questionIndex++;
    } else {
        alert("Interview Completed! Please click Stop.");
        nextQuestionBtn.disabled = true;
    }
};

stopBtn.onclick = async () => {
    stopCurrentRecording();
    clearTimeout(randomScreenshotTimeout);
    nextQuestionBtn.disabled = true;
    startBtn.disabled = false;

    await fetch('/stop_camera', { method: 'POST' });
    document.getElementById('cameraFeed').src = '';

    try {
        const res = await fetch(`/export_pdf?session_id=${sessionId}`);
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "interview_report.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
    } catch (err) {
        console.error("PDF export failed", err);
    }

    if (ws) ws.close(); // Close WebSocket
};
