import cv2
import sys
from flask import Flask, Response

app = Flask(__name__)

# Get camera index from command line arguments, default to 0
camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Initialize the camera
camera = cv2.VideoCapture(camera_index)

def generate_frames():
    while True:
        success, frame = camera.read()  # Read the frame from the camera
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
        <h1>Webcam Feed</h1>
        <img src="/video_feed" />
    '''

if __name__ == '__main__':
    if not camera.isOpened():
        print(f"Camera with index {camera_index} not found.")
        sys.exit(1)
        
    app.run(host='0.0.0.0', port=5000)
