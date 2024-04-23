from langchain.llms import CTransformers

llm = CTransformers(model='Thebloke/Llama-2-7B-Chat-GGML',
                    model_type='llama',
                    config={'max_new_tokens':256,
                            'temperature': 0.01})
pred = llm.predict("hi, wh  is mango")
print(pred)

































from flask import Flask, render_template, request, jsonify
from camera_detect import detect_facial_emotion
from speech_detect import detect_speech_emotion
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_gpt2_response(text, sentiment, emotion):
    prompt = f"As a personal assistant, I'm here to help you based on the information you provided. Let's see how I can assist you:\n\nSpeech Text: {text}\nDetected Sentiment: {sentiment}\nDetected Emotion: {emotion}\n\nConsidering the context of your speech and the detected sentiment and emotion, here's my thoughtful response:\n"

    # Set the pad_token to the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.encode_plus(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()

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

    # Generate response using GPT-2
    response = generate_gpt2_response(spoken_text, sentiment, detected_emotion)
    print(response,'GPT-Response-------------')

    return jsonify({
        'response': response
    })

if __name__ == '__main__':
    app.run(debug=True)