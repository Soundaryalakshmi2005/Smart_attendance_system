from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import pandas as pd
from deepface import DeepFace
from datetime import datetime

app = Flask(__name__)

# Path to stored images for recognition
stored_images = {
    "Soundarya": r"C:\HTML\system\images\soundarya.jpg",
    "Sakthi": r"C:\HTML\system\images\sakthi.jpg"
}

attendance_file = "attendance.csv"

# Ensure attendance file exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Roll Number", "Date", "Time", "Status"])
    df.to_csv(attendance_file, index=False)

# Function to capture video stream
def generate_frames():
    cap = cv2.VideoCapture(0)  # Change to 1 if 0 doesn't work

    if not cap.isOpened():
        print("Error: Could not access webcam")
        return  

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame")
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to recognize face
def recognize_face():
    cap = cv2.VideoCapture(0)  # Ensure consistent camera index

    if not cap.isOpened():
        return "Error: Could not access webcam"
    
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        return "Error: Could not capture image"
    
    cv2.imwrite("captured.jpg", frame)
    cap.release()  # Release webcam after capturing
    cv2.destroyAllWindows()
    
    for name, img_path in stored_images.items():
        try:
            result = DeepFace.verify("captured.jpg", img_path, model_name='VGG-Face')
            if result["verified"]:
                mark_attendance(name)
                return f"Attendance marked for {name}"
        except Exception as e:
            return f"Error in face recognition: {str(e)}"
    
    return "No match found"

# Function to mark attendance
def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    now = datetime.now()
    
    new_entry = pd.DataFrame([{
        "Name": name,
        "Roll Number": "Unknown",
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": "Present"
    }])
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)

@app.route('/')
def index():
    df = pd.read_csv(attendance_file)
    return render_template('index.html', attendance_data=df.to_dict(orient='records'))

@app.route('/capture', methods=['POST'])
def capture():
    result = recognize_face()
    return jsonify({"message": result})

@app.route('/get_attendance')
def get_attendance():
    df = pd.read_csv(attendance_file)
    return jsonify({"attendance": df.values.tolist()})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Enable threading for smoother webcam access
