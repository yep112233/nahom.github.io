/**
 * Pose Detection Application
 * Using TensorFlow.js and Teachable Machine
 * Created: January 2024
 */

// Model URL from Teachable Machine
//**************************************************
//* as before, paste your lnk below
let URL = "https://teachablemachine.withgoogle.com/models/CMBhs4EAW/";




let model, webcam, ctx, labelContainer, maxPredictions;

// Dynamic pose tracking
let poseStates = {};
let explosionActive = false;
let explosionSound = new Audio('explsn.mp3');

function setModelURL(url) {
    URL = url;
    // Reset states when URL changes
    poseStates = {};
    explosionActive = false;
}

/**
 * Initialize the application
 */
async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    const video = document.getElementById('instructionVideo');
    video.volume = 0.4;

    try {
        model = await tmPose.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        const width = 600;
        const height = 600;
        const flip = true;
        webcam = new tmPose.Webcam(width, height, flip);
        await webcam.setup();
        await webcam.play();
        window.requestAnimationFrame(loop);

        const canvas = document.getElementById("canvas");
        canvas.width = width;
        canvas.height = height;
        ctx = canvas.getContext("2d");
        labelContainer = document.getElementById("label-container");
        for (let i = 0; i < maxPredictions; i++) {
            labelContainer.appendChild(document.createElement("div"));
        }
    } catch (error) {
        console.error("Error initializing model:", error);
    }
}

async function loop(timestamp) {
    webcam.update();
    await predict();
    window.requestAnimationFrame(loop);
}

function playExplosionSound() {
    const newSound = new Audio('explsn.mp3');
    newSound.volume = 1.0;
    newSound.play();
}

async function predict() {
    try {
        const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
        const prediction = await model.predict(posenetOutput);
        const video = document.getElementById('instructionVideo');

        for (let i = 0; i < maxPredictions; i++) {
            const classPrediction =
                prediction[i].className + ": " + prediction[i].probability.toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;

            // Check pose dynamically
            checkPose(prediction[i], video);
        }

        drawPose(pose, explosionActive);
    } catch (error) {
        console.error("Error in predict:", error);
    }
}

function checkPose(prediction, video) {
    const time = video.currentTime;
    const prob = prediction.probability;

    // Only respond to pose1 through pose5 labels
    const poseNumber = prediction.className.toLowerCase().replace(/[^0-9]/g, '');
    const isPoseLabel = prediction.className.toLowerCase().includes('pose') && poseNumber >= 1 && poseNumber <= 5;

    if (!isPoseLabel) return;

    if (!poseStates[`pose${poseNumber}`]) {
        poseStates[`pose${poseNumber}`] = {
            triggered: false,
            firstWindowTriggered: false,
            secondWindowTriggered: false
        };
    }

    if (prob > 0.8 && !explosionActive) {
        const poseState = poseStates[`pose${poseNumber}`];

        switch(poseNumber) {
            case '1':
                if (time >= 0.9 && time <= 3.0 && !poseState.triggered) {
                    triggerExplosion(poseState);
                }
                break;
            case '2':
                if (time >= 5.5 && time <= 7.5 && !poseState.triggered) {
                    triggerExplosion(poseState);
                }
                break;
            case '3':
                if ((time >= 11.5 && time <= 13.0 && !poseState.firstWindowTriggered) ||
                    (time >= 17.5 && time <= 19.5 && !poseState.secondWindowTriggered)) {
                    if (time <= 13.0) {
                        poseState.firstWindowTriggered = true;
                    } else {
                        poseState.secondWindowTriggered = true;
                    }
                    explosionActive = true;
                    playExplosionSound();
                    setTimeout(() => { explosionActive = false; }, 300);
                }
                break;
            case '4':
                if (time >= 15.5 && time <= 16.6 && !poseState.triggered) {
                    triggerExplosion(poseState);
                }
                break;
            case '5':
                if (time >= 19.5 && !poseState.triggered) {
                    triggerExplosion(poseState);
                }
                break;
        }
    }
}

function triggerExplosion(poseState) {
    explosionActive = true;
    poseState.triggered = true;
    playExplosionSound();
    setTimeout(() => { explosionActive = false; }, 300);
}

function drawPose(pose, explode) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        if (pose) {
            const minPartConfidence = 0.5;
            if (explode) {
                pose.keypoints.forEach(keypoint => {
                    if (keypoint.score > minPartConfidence) {
                        const scale = 3;
                        ctx.beginPath();
                        ctx.arc(keypoint.position.x, keypoint.position.y, 10 * scale, 0, 2 * Math.PI);
                        ctx.fillStyle = '#FF0000';
                        ctx.fill();
                    }
                });
            } else {
                tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
                tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
            }
        }
    }
}

async function playInstructionVideo() {
    const video = document.getElementById('instructionVideo');
    const videoSrc = video.getAttribute('data-video-src') || 'vid.mp4';
    video.src = videoSrc;
    const videoContainer = video.parentElement;

    video.addEventListener('timeupdate', () => {
        const minutes = Math.floor(video.currentTime / 60);
        const seconds = Math.floor(video.currentTime % 60);
        document.getElementById('videoTime').textContent = 
            `Time: ${minutes}:${seconds.toString().padStart(2, '0')}`;
    });

    const videoCanvas = document.createElement('canvas');
    videoCanvas.id = 'poseCanvas';
    videoCanvas.style.position = 'absolute';
    videoCanvas.style.left = '0';
    videoCanvas.style.top = '0';
    videoCanvas.width = 600;
    videoCanvas.height = 450;

    videoContainer.style.position = 'relative';
    videoContainer.appendChild(videoCanvas);
    const videoCtx = videoCanvas.getContext('2d');

    video.play();

    async function processFrame() {
        if (!video.paused && !video.ended) {
            try {
                const { pose, posenetOutput } = await model.estimatePose(video);
                videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);

                if (pose) {
                    tmPose.drawKeypoints(pose.keypoints, 0.6, videoCtx);
                    tmPose.drawSkeleton(pose.keypoints, 0.6, videoCtx);
                }
            } catch (error) {
                console.error('Pose detection error:', error);
            }
            requestAnimationFrame(processFrame);
        }
    }

    if (model) {
        processFrame();
    } else {
        console.log("https://teachablemachine.withgoogle.com/models/CMBhs4EAW/");
    }
}

function stopInstructionVideo() {
    const video = document.getElementById('instructionVideo');
    video.pause();
    video.currentTime = 0;
    const canvas = video.parentElement.querySelector('canvas');
    if (canvas) {
        canvas.remove();
    }
    pose1Triggered = false;
    pose2Triggered = false;
    pose3FirstWindowTriggered = false;
    pose3SecondWindowTriggered = false;
    pose4Triggered = false;
    pose5Triggered = false;
}

function stopWebcam() {
    if (webcam) {
        webcam.stop();
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}