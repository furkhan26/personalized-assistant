import os
import logging
import speech_recognition as sr
from textblob import TextBlob

def categorize_sentiment(sentiment_polarity):
    if sentiment_polarity >= 0.8:
        return 'Very Happy'
    elif sentiment_polarity >= 0.6:
        return 'Happy'
    elif sentiment_polarity >= 0.4:
        return 'Mildly Happy'
    elif sentiment_polarity >= 0.2:
        return 'Slightly Happy'
    elif sentiment_polarity > -0.2:
        return 'Neutral'
    elif sentiment_polarity > -0.4:
        return 'Slightly Sad'
    elif sentiment_polarity > -0.6:
        return 'Mildly Sad'
    elif sentiment_polarity > -0.8:
        return 'Sad'
    else:
        return 'Very Sad'

def detect_speech_emotion():
    os.makedirs('logs/speech_logs', exist_ok=True)
    logging.basicConfig(filename='logs/speech_logs/speech_recognition.log', level=logging.INFO)

    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            logging.info("Listening...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            logging.info("Speech Recognized: {}".format(text))
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_label = categorize_sentiment(sentiment_polarity)
            print("Sentiment Label:", sentiment_label)
            logging.info("Sentiment Label: {}".format(sentiment_label))

            return text, sentiment_label

        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            logging.error("Speech Recognition Error: Could not understand audio.")
        except sr.RequestError as e:
            print("Error fetching results; {0}".format(e))
            logging.error("Speech Recognition Error: {0}".format(e))

    except Exception as ex:
        print("An error occurred: {0}".format(ex))
        logging.exception("An error occurred: {0}".format(ex))

    return None, None


if __name__ == "__main__":
    detect_speech_emotion()

# text, sentiment = detect_speech_emotion()
# print("Detected Text:", text)
# print("Sentiment Label:", sentiment)
