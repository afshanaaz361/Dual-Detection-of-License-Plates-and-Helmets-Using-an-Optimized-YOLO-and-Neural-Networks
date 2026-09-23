from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify  ,Response
import os
from ultralytics import YOLO
import cv2
from werkzeug.utils import secure_filename
import mimetypes
import subprocess
import random




app = Flask(__name__)

# Set upload and output directories
UPLOAD_FOLDER = './static/uploads'
DETECTION_FOLDER = './static/detections'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

# Load YOLO model
model = YOLO(r'D:/SHIVA_2025_MINI/OWN/End-to-End-Sign-Language-Detection-Project-main/teja_copy/best.pt')

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)
















@app.route("/")
def index():
    return render_template('index.html')

@app.route("/login")
def login():
    return render_template('sign-in.html')

@app.route("/sigin", methods=['POST', 'GET'])
def login_data():
    if request.method == "POST":
        email = request.form["email"]
        pswd = request.form["password"]
        if email == "admin" and pswd == "admin":
            return render_template("home.html")
    return render_template('sign-in.html')

@app.route("/home")
def home():
    return render_template('home.html')






@app.route("/image")
def image():
    return render_template('image.html')

@app.route('/prediction1', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No file selected', 400

    # Save the uploaded image
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    # Output image path
    output_path = os.path.join(app.config['DETECTION_FOLDER'], f'detected_{file.filename}')

    # Load and detect objects in the image
    image = cv2.imread(input_path)
    results = model.predict(image)

    # Generate unique colors for each class
    num_classes = len(model.names)
    random.seed(42)  # For consistent coloring
    colors = {cls_id: [random.randint(0, 255) for _ in range(3)] for cls_id in range(num_classes)}

    # Annotate detections and count objects
    detections = results[0].boxes.data.tolist()
    class_counts = {model.names[i]: 0 for i in range(num_classes)}  # Initialize counts for each class

    # Sort detections by confidence (higher confidence drawn last)
    detections.sort(key=lambda x: x[4], reverse=True)

    for result in detections:
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Bounding box coordinates
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Label with confidence

        # Update count for the detected class
        class_counts[model.names[int(cls)]] += 1

        # Get color for the class
        color = colors[int(cls)]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=4)

        # Adjust label position to avoid overlap with other boxes
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

        # Draw label background
        cv2.rectangle(image, (x1, label_y - label_size[1] - 5), (x1 + label_size[0] + 5, label_y + 5), color, -1)

        # Put label text
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

    # Save the annotated image
    cv2.imwrite(output_path, image)

    # Prepare class counts for display
    class_count_display = {k: v for k, v in class_counts.items() if v > 0}  # Show only detected classes

    # Display the result page with the uploaded and detected images and counts
    return render_template(
        'image.html', 
        input_image=url_for('static', filename=f'uploads/{file.filename}'),
        output_image=url_for('static', filename=f'detections/detected_{file.filename}'),
        class_counts=class_count_display
    )










@app.route("/video")
def video():
    return render_template('video.html')


ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'flv', 'webm', 'wmv', 'mpeg', '3gp'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/prediction2', methods=['POST'])
def predict_video():
    """Handle video upload and process with YOLO."""
    if 'video' not in request.files:
        return redirect(url_for('home'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('home'))

    if not allowed_file(file.filename):
        return 'Invalid file type. Allowed types are: mp4, mov, avi, mkv, flv, webm, wmv, mpeg, 3gp', 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video formats
        result_video_path = process_video(file_path, filename)
        return render_template('video.html', 
                                input_video=filename, 
                                output_video=result_video_path)

@app.route('/prediction2/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  

@app.route('/prediction2/videos/<filename>')
def result_video(filename):
    return send_from_directory(app.config['DETECTION_FOLDER'], filename, mimetype='video/mp4')

     

def process_video(input_path, filename):
    # OpenCV Video Capture
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['DETECTION_FOLDER'], f"result_{filename}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Temporary file for processed frames
    temp_output = os.path.join(app.config['DETECTION_FOLDER'], "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Temporary OpenCV video
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Processing: Annotate the frame with detections
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # FFmpeg Encoding for Final Output
    ffmpeg_command = [
        "ffmpeg", "-y",  # Overwrite output if exists
        "-i", temp_output,   # Input temporary video
        "-c:v", "libx264",   # H.264 codec for encoding
        "-preset", "medium", # Encoding speed
        "-crf", "23",        # Quality (Lower is better)
        "-c:a", "aac",       # Audio codec
        "-strict", "experimental",
        output_path          # Final output video
    ]

    try:
        # Run FFmpeg to re-encode
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    # Remove temporary file
    os.remove(temp_output)

    return f"result_{filename}"  # Return relative path















# Global variable to control the video feed
camera = None

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)  # Open the default camera

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform YOLO detection
        results = model.predict(frame, verbose=False)

        # Annotate detections
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Bounding box coordinates
            label = f"{model.names[int(cls)]} {conf:.2f}"  # Label with confidence

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame to browser
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
    return render_template('camera.html')  # Ensure your HTML file is named correctly



@app.route("/performance")
def performance():
    return render_template("performance.html")

@app.route("/charts")
def charts():
    return render_template('charts.html')

@app.route("/logout")
def logout():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)