from flask import Flask, render_template, Response, request
import cv2
from deepface import DeepFace
import numpy as np
import joblib

app = Flask(__name__)
import requests

YOUTUBE_API_KEY = "AIzaSyDYEeSTrT7pPpVzpmaJ491gxogVxfWwpvM"

def fetch_youtube_videos(query, max_results=12):
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'key': YOUTUBE_API_KEY,
        'maxResults': max_results
    }
    response = requests.get(url, params=params)
    videos = []
    if response.status_code == 200:
        data = response.json()
        for item in data['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'title': video_title})
    return videos


# Load the trained model and pre-fitted scaler
model = joblib.load("train_model.pkl")
scaler = joblib.load("scaler.pkl")

# Global variable to store emotion
detected_emotion = "No Emotion Detected"

# Camera setup
camera = cv2.VideoCapture(0)

def generate_frames():
    global detected_emotion
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                detected_emotion = result[0]['dominant_emotion']

                # Draw rectangle and emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error analyzing face: {e}")
        
        # Encode the frame as a JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        openness = float(request.form['openness'])
        neuroticism = float(request.form['neuroticism'])
        conscientiousness = float(request.form['conscientiousness'])
        agreeableness = float(request.form['agreeableness'])
        extraversion = float(request.form['extraversion'])

        input_data = [gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]
        scaled_data = scaler.transform(np.array(input_data).reshape(1, -1))
        personality = model.predict(scaled_data)[0]

        global detected_emotion
        emotion = detected_emotion

        # Fetch video recommendations
        if emotion =='anger':
            query = f"{'patience videos'}"
        elif emotion == 'disgust':
            query = f"{'love songs'}"
        elif emotion == 'fear':
            query = f"{'devotion songs'}"
        elif emotion == 'happy':
            query = f"{'comedy skits'}"
        elif emotion == 'sad':
            query = f"{'funnyy videos'}"
        elif emotion=='contempt':
            query = f"{'respectfull videos'}"
        elif emotion=='neutral':
            query = f"{'recent songs'}"
        elif emotion=='suprise':
            query = f"{'sentiment family songs'}"
        else:
            query = f"{personality}"
        query = query+' in hindi'

        videos = fetch_youtube_videos(query)

        return render_template('result.html', emotion=emotion, personality=personality, videos=videos)
    return render_template('prediction.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)