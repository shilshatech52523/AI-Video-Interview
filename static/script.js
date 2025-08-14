const videoElement = document.getElementById("preview");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const nextQuestionBtn = document.getElementById("nextQuestionBtn");
const warningDiv = document.getElementById("warningMsg");

let mediaRecorder;
let recordedChunks = [];
let questionIndex = 0;
let currentQuestion = "";
let sessionId = "";
let randomScreenshotTimeout = null;
let extraPersonCount = 0;

// Questions
const questions = [
    "Q1: What is your name?",
    "Q2: Where do you live?",
    "Q3: What is your favorite programming language?",
    "Q4: Tell me about your hobbies.",
    "Q5: What is your goal for this year?"
];

// WebSocket for cheating detection
let ws;
function initWebSocket() {
    ws = new WebSocket("ws://localhost:8000/cheating_status");
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.cheating) {
            stopRecordingAndClose("⚠ Cheating detected! Recording stopped.");
        }
    };
    ws.onclose = () => console.log("WebSocket closed");
}

// Logs
function logStatus(message) {
    const logDiv = document.getElementById("statusLog");
    const time = new Date().toLocaleTimeString();
    logDiv.innerHTML += `<div>[${time}] ${message}</div>`;
}

// Warning messages
function showWarning(message) {
    warningDiv.textContent = message;
    warningDiv.style.opacity = '1';
    setTimeout(() => {
        warningDiv.style.transition = "opacity 1s";
        warningDiv.style.opacity = '0';
    }, 4000);
}

// FaceMesh setup
const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
faceMesh.setOptions({ maxNumFaces: 5, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
faceMesh.onResults(onResults);

// Draw rectangles
function drawFaceBoundingBoxes(allLandmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const colors = ['#00FF00', '#FF0000', '#0000FF', '#FFA500', '#FF00FF'];

    allLandmarks.forEach((landmarks, idx) => {
        const xs = landmarks.map(lm => lm.x * canvas.width);
        const ys = landmarks.map(lm => lm.y * canvas.height);

        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        ctx.strokeStyle = colors[idx % colors.length];
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.rect(minX, minY, maxX - minX, maxY - minY);
        ctx.stroke();
    });
}

// onResults
function onResults(results) {
    const faces = results.multiFaceLandmarks || [];

    if(faces.length === 0){
        showWarning("No face detected. Please show your face clearly.");
        ctx.clearRect(0,0,canvas.width,canvas.height);
        return;
    }

    drawFaceBoundingBoxes(faces);

    if(faces.length > 1){
        extraPersonCount++;
        showWarning("Multiple faces detected! Only one person allowed.");
        logStatus(`Extra face/device detected: ${extraPersonCount}/3`);

        if(extraPersonCount >= 3){
            stopRecordingAndClose("⚠ 3 detections of extra person/device. Session closed!");
        }
    }
}

// Camera init
async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:true });
    videoElement.srcObject = stream;
    return new Promise((resolve)=>{
        videoElement.onloadedmetadata = ()=>{
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const camera = new Camera(videoElement,{
                onFrame: async ()=> await faceMesh.send({image: videoElement}),
                width: videoElement.videoWidth,
                height: videoElement.videoHeight
            });
            camera.start();
            resolve(stream);
        }
    });
}

// TTS
function speak(text){
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang="en-US";
    utterance.rate=1;
    utterance.pitch=1;
    window.speechSynthesis.speak(utterance);
}

// Screenshots
function captureScreenshot(){
    const offCanvas=document.createElement("canvas");
    offCanvas.width=videoElement.videoWidth;
    offCanvas.height=videoElement.videoHeight;
    const ctx2 = offCanvas.getContext("2d");
    ctx2.drawImage(videoElement,0,0,offCanvas.width,offCanvas.height);

    offCanvas.toBlob(async (blob)=>{
        const formData = new FormData();
        formData.append("file", blob,"screenshot.jpg");
        await fetch("/save_screenshot",{method:"POST",body:formData});
        showWarning("📸 Screenshot taken!");
    },"image/jpeg",0.95);

    scheduleNextScreenshot();
}

function scheduleNextScreenshot(){
    const interval = Math.floor(Math.random()*5000)+5000;
    randomScreenshotTimeout = setTimeout(captureScreenshot,interval);
}

// Recording
async function startRecording(questionText){
    const stream = await initCamera();
    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (event)=>{
        if(event.data.size>0) recordedChunks.push(event.data);
    }

    mediaRecorder.onstop= async ()=>{
        if(recordedChunks.length>0){
            const blob = new Blob(recordedChunks,{type:"video/webm"});
            const formData = new FormData();
            formData.append("file",blob,"recording.webm");
            formData.append("question",questionText);
            formData.append("session_id",sessionId);
            await fetch("/upload",{method:"POST",body:formData});
        }
    }

    mediaRecorder.start();
    speak(questionText);
    captureScreenshot();
}

function stopCurrentRecording(){
    if(mediaRecorder && mediaRecorder.state!=="inactive") mediaRecorder.stop();
}

// // Stop & close on 3 detections or cheating
// async function stopRecordingAndClose(message){
//     stopCurrentRecording();
//     clearTimeout(randomScreenshotTimeout);
//     nextQuestionBtn.disabled=true;
//     startBtn.disabled=false;

//     alert(message);

//     if(ws) ws.close();
// }

async function stopRecordingAndClose(message){
    // 1️⃣ Final screenshot before stopping
    const offCanvas = document.createElement("canvas");
    offCanvas.width = videoElement.videoWidth;
    offCanvas.height = videoElement.videoHeight;
    const ctx2 = offCanvas.getContext("2d");
    ctx2.drawImage(videoElement, 0, 0, offCanvas.width, offCanvas.height);

    offCanvas.toBlob(async (blob)=>{
        const formData = new FormData();
        formData.append("file", blob,"final_screenshot.jpg");
        await fetch("/save_screenshot",{method:"POST",body:formData});
        console.log("Final screenshot taken before closing session.");
    }, "image/jpeg", 0.95);

    // 2️⃣ Stop recording
    stopCurrentRecording();
    clearTimeout(randomScreenshotTimeout);
    nextQuestionBtn.disabled = true;
    startBtn.disabled = false;

    // 3️⃣ Show alert and close WebSocket
    alert(message);
    if(ws) ws.close();
}


// Button handlers
startBtn.onclick=()=>{
    initWebSocket();
    sessionId = crypto.randomUUID();
    questionIndex=0;
    extraPersonCount=0;

    startBtn.disabled=true;
    nextQuestionBtn.disabled=false;
    stopBtn.disabled=false;

    currentQuestion = questions[questionIndex];
    startRecording(currentQuestion);

    questionIndex++;
}

nextQuestionBtn.onclick=()=>{
    stopCurrentRecording();
    if(questionIndex<questions.length){
        currentQuestion = questions[questionIndex];
        startRecording(currentQuestion);
        questionIndex++;
    }else{
        alert("Interview Completed! Please click Stop.");
        nextQuestionBtn.disabled=true;
    }
}

stopBtn.onclick=async()=>{
    stopCurrentRecording();
    clearTimeout(randomScreenshotTimeout);
    nextQuestionBtn.disabled=true;
    startBtn.disabled=false;

    if(ws) ws.close();
    alert("Interview session stopped.");
}