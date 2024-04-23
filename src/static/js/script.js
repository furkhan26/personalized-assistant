// Get the video and canvas elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// Start the video stream when the "Ask Assistant" button is clicked
document.getElementById('askAssistantBtn').addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        await new Promise(resolve => setTimeout(resolve, 1000));
        const frameDataURL = captureFrame();
        const response = await fetch('/detect_emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frameData: frameDataURL
            })
        });

        const data = await response.json();
        const detectedEmotion = data.detectedEmotion;

        stream.getTracks().forEach(track => track.stop());

        document.getElementById('speakNowPrompt').style.display = 'block';

        const speechData = await detectSpeechEmotion();
        await generateResponse(speechData.text, speechData.sentiment, detectedEmotion);
    } catch (error) {
        console.error('Error:', error);
    }
});

// Capture a frame from the video stream
function captureFrame() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
    return dataURL;
}

// Detect speech emotion using the microphone
async function detectSpeechEmotion() {
    const response = await fetch('/detect_speech_emotion', {
        method: 'POST'
    });

    const data = await response.json();
    return data;
}
async function generateResponse(spokenText, sentiment, detectedEmotion) {
    const response = await fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            spokenText: spokenText,
            sentiment: sentiment,
            detectedEmotion: detectedEmotion
        })
    });

    const data = await response.json();
    document.getElementById('response').textContent = data.response;
    document.getElementById('speakNowPrompt').style.display = 'none';
}