from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, Response
import os
import sqlite3
import cv2
import subprocess
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Database Connection Function
def get_db_connection():
    conn = sqlite3.connect('Python.db')
    conn.row_factory = sqlite3.Row  # Dictionary-like rows
    return conn

# Create Users Table
def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Ensure Table Exists
create_users_table()

# Folder for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load YOLO Model
model = YOLO(r'C:/Users/AFSHA NAAZ/Desktop/VTPAI03/runs/detect/train2/weights/best.pt')

@app.route("/")
def index():
    return render_template('index.html')

# SIGNUP
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO Users (username, email, password) VALUES (?, ?, ?)", 
                           (username, email, password))
            conn.commit()
            return redirect(url_for("login"))  # Redirect to login after signup
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already exists")
        finally:
            conn.close()

    return render_template('signup.html')

# LOGIN
@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Users WHERE email=? AND password=?", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["email"] = user["email"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template('login.html')

# HOME PAGE (After login)
@app.route("/home")
def home():
    if "email" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", username=session.get("username"))



@app.route("/result")
def result():
    return render_template('image.html')

@app.route("/video")
def video():
    return render_template('video.html')

# IMAGE & VIDEO PROCESSING
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video processing
            result_video_path = process_video(file_path, filename)
            return render_template('image.html', 
                                   original_file=filename, 
                                   result_video=result_video_path)

        else:  # Image processing
            results = model(file_path)
            result_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            annotated_frame = results[0].plot()
            cv2.imwrite(result_img_path, annotated_frame)
            return render_template('image.html', 
                                   original_file=filename, 
                                   result_file=f"result_{filename}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')

def process_video(input_path, filename):
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    temp_output = os.path.join(app.config['RESULTS_FOLDER'], "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Convert to proper video format using FFmpeg
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", temp_output, 
        "-c:v", "libx264", "-preset", "medium", 
        "-crf", "23", "-c:a", "aac", "-strict", "experimental",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    os.remove(temp_output)

    return f"result_{filename}"

# LIVE CAMERA FEED
camera = None

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        results = model.predict(frame, verbose=False)

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{model.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close_camera', methods=['POST'])
def close_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return '', 204

@app.route('/camera')
def camera_page():
    return render_template('camera.html')
@app.route("/details")
def details():
    if 'email' not in session:
        return redirect('/login')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Users')
    result = cursor.fetchall()
    conn.close()
    
    return render_template('details.html', result=result)





if __name__ == '__main__':
    app.run(debug=True)
