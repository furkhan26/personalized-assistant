from transformers import GPT2LMHeadModel, GPT2Tokenizer
from camera_detect import detect_facial_emotion
from speech_detect import detect_speech_emotion

model_name = "gpt2-large"
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
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    detected_emotion = detect_facial_emotion()
    print('detected_emotion: ',detected_emotion)
    text, sentiment = detect_speech_emotion()

    if text and sentiment and detected_emotion:
        gpt2_response = generate_gpt2_response(text, sentiment, detected_emotion)
        print("GPT-2 Response:", gpt2_response)
    else:
        print("Failed to generate a response due to missing data.")