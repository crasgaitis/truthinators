import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading
import time

model = tf.keras.models.load_model("model_baseline_dense2_128.keras")

emotion_label_to_text = {
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 
    4: 'sadness', 5: 'surprise', 6: 'neutral'
}

pos_emotion = ['happiness', 'surprise']
neg_emotion = ['anger', 'disgust', 'fear', 'sadness']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = sr.Recognizer()
analyzer = SentimentIntensityAnalyzer()

detected_emotion = "neutral"
detected_sentiment = "neutral"
audio_text = "placeholder"
audio_lock = threading.Lock()

def detect_emotion_and_audio(frame):
    global detected_emotion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(face)
        emotion_idx = np.argmax(prediction)
        detected_emotion = emotion_label_to_text[emotion_idx]
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

def listen_to_audio():
    global audio_text, detected_sentiment
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio_data = recognizer.listen(source)
                audio_text = recognizer.recognize_google(audio_data)
                sentiment_score = analyzer.polarity_scores(audio_text)
                if sentiment_score['compound'] >= 0.0001:
                    detected_sentiment = "positive"
                elif sentiment_score['compound'] <= -0.000001:
                    detected_sentiment = "negative"
                else:
                    detected_sentiment = "neutral"
            except sr.UnknownValueError:
                audio_text = "Unable to understand audio"
                detected_sentiment = "neutral"
            except sr.RequestError:
                audio_text = "Error with speech recognition service"
                detected_sentiment = "neutral"
                
            if detected_sentiment == 'positive':
                if detected_emotion in pos_emotion:
                    truth_result.write(f"Truth!")
                else:
                    truth_result.write(f"Lie!")
            elif detected_sentiment == 'negative':
                if detected_emotion in neg_emotion:
                    truth_result.write(f"Truth!")
                else:
                    truth_result.write(f"Lie!")
            else:
                if detected_emotion=='neutral':
                    truth_result.write(f"Truth!")
                else:
                    truth_result.write(f"Lie!")
            time.sleep(0.5)

st.title("Real-Time Video, Audio Text, and Model Predictions")

video_placeholder = st.empty()
text_placeholder2 = st.empty()
text_placeholder = st.empty()
sentiment_placeholder = st.empty()

truth_result = st.empty()

audio_thread = threading.Thread(target=listen_to_audio, daemon=True)
audio_thread.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect_emotion_and_audio(frame)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    text_placeholder2.write(f"Recognized Video: {detected_emotion}")
    
    text_placeholder.write(f"Recognized Text: {audio_text}")
    sentiment_placeholder.write(f"Sentiment: {detected_sentiment}")

        
    time.sleep(1)

cap.release()
