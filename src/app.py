from flask import Flask, render_template, request, jsonify
from camera_detect import detect_facial_emotion
from speech_detect import detect_speech_emotion
from langchain.llms import CTransformers

app = Flask(__name__)


def llama_resp(text, sentiment, emotion):
    prompt = f"As a personal assistant, I'm here to help you based on the information you provided. Let's see how I can assist you:\n\nSpeech Text: {text}\nDetected Sentiment: {sentiment}\nDetected Emotion: {emotion}\n\nConsidering the context of your speech and the detected sentiment and emotion, here's my thoughtful response:\n"

    llm = CTransformers(model='Thebloke/Llama-2-7B-Chat-GGML',
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature': 0.01})
    response = llm.predict(prompt)

    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Detect facial emotion from the captured frame
    detected_emotion = detect_facial_emotion()

    return jsonify({
        'detectedEmotion': detected_emotion
    })

@app.route('/detect_speech_emotion', methods=['POST'])
def detect_speech_emotion_route():
    text, sentiment = detect_speech_emotion()

    return jsonify({
        'text': text,
        'sentiment': sentiment
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    spoken_text = data['spokenText']
    sentiment = data['sentiment']
    detected_emotion = data['detectedEmotion']

    response = llama_resp(spoken_text, sentiment, detected_emotion)
    print(response,'Llama-Response-------------')

    return jsonify({
        'response': response
    })

if __name__ == '__main__':
    app.run(debug=True)