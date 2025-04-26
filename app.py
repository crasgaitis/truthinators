import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
import numpy as np
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import random
import time
import os
import wave
import pyaudio
import serial 

# TODO: Test arduino connection
baud = 9600
port = "COM9"
shocker = serial.Serial(port, baud)

os.makedirs("temp_files", exist_ok=True)
VIDEO_PATH = "temp_files/input.mp4"
AUDIO_PATH = "temp_files/input.wav"

# TODO: CNN Model
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

# Globals
is_running = False
video_writer = None
frames_list = []

# GUI setup
root = tk.Tk()
root.title("Lie Detector")

status_label = Label(root, text="Click Start", font=("Arial", 18))
status_label.pack(pady=10)

video_label = Label(root)
video_label.pack()

transcript_display = Label(root, text="", font=("Arial", 12), wraplength=600, justify="left")
transcript_display.pack(pady=5)

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(face, verbose=0)
        emotion_idx = np.argmax(prediction)
        return emotion_label_to_text[emotion_idx]
    return "neutral"

def record_video():
    global video_writer, frames_list
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (width, height))
    while is_running:
        ret, frame = cap.read()
        if ret:
            video_writer.write(frame)
            frames_list.append(frame)
    cap.release()
    video_writer.release()

def record_audio():
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1
    rate = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=fmt, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []
    while is_running:
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(AUDIO_PATH, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(fmt))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def analyze():
    # 1. Transcript
    transcript = ""
    with sr.AudioFile(AUDIO_PATH) as source:
        audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
        except:
            transcript = "[Unable to transcribe]"

    # 2. TODO: Sentiment. Replace with speech model.
    sensitivity_factor = 1
    scaled_sens_fact = 0.25 (sensitivity_factor ** 2) + 0.75 * sensitivity_factor + 1
    sentiment_score = analyzer.polarity_scores(transcript)
    if sentiment_score['compound'] >= 0.0001 / scaled_sens_fact:
        sentiment = "positive"
    elif sentiment_score['compound'] <= -0.000001 / scaled_sens_fact:
        sentiment = "negative"
    else:
        sentiment = "neutral"

   # 3. Video data: random frame and detect emotion
    chosen_frame = random.choice(frames_list)
    chosen_emotion = detect_emotion(chosen_frame)

    # 4. TODO: Determine lie/truth. Add a sensitivity metric. 
    # sensitivity = 1
    # shock_str = 1
    # if sentiment_score['compound'] >= (sensitivity * 0.0001):
    #     shock_str = 2
    # elif sentiment_score['compound'] <= (sensitivity * -0.000001):
    #     shock_str = 2
    # else:
    #     shock_str = 1
    
    # With current setup, consider editing the ranges in line 113, 115.
    # With future setup, consider adapting heuristics in new_app.py    
    if sentiment == "positive" and chosen_emotion in pos_emotion:
        final_result = "Truth"
    elif sentiment == "negative" and chosen_emotion in neg_emotion:
        final_result = "Truth"
    elif sentiment == "neutral" and chosen_emotion == "neutral":
        final_result = "Truth"
    else:
        final_result = "Lie"

    # Display image
    rgb = cv2.cvtColor(chosen_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update GUI
    status_label.config(text=f"Result: {final_result}")
    transcript_display.config(text="Transcript:\n" + transcript)
    
    if final_result == "Lie":
        cmd = "H"
        cmd2 = "L"
        sleep_time = (5-1)(np.abs(sentiment_score)) + 1
        print("Sleep time: " + sleep_time)
        # Send cmd message to the car
        # Right now we don't really need to handle the ack message but in the future could support better error handling
        shocker.write(cmd.encode())
        shocker.flush() # make sure it all sends before you start reading
        print("Sent: " + cmd)
        time.sleep(sleep_time)
        shocker.write(cmd2.encode())
        shocker.flush()
        print("Sent: " + cmd2)

def start_recording():
    global is_running, frames_list
    is_running = True
    frames_list.clear()
    status_label.config(text="Analyzing...")
    threading.Thread(target=record_video, daemon=True).start()
    threading.Thread(target=record_audio, daemon=True).start()
    start_btn.config(text="Stop", command=stop_recording)

def stop_recording():
    global is_running
    is_running = False
    time.sleep(2)
    analyze()
    start_btn.config(text="Start", command=start_recording)

start_btn = Button(root, text="Start", command=start_recording, font=("Arial", 16))
start_btn.pack(pady=10)

root.mainloop()