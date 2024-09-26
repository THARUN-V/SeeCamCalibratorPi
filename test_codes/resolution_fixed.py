import cv2
import sys
from flask import Flask, Response, request, render_template_string

app = Flask(__name__)

# Get camera index from command line arguments, default to 0
camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Initialize the camera
camera = cv2.VideoCapture(camera_index)

# Set desired resolution (e.g., 1280x720)
desired_width = 960
desired_height = 540
camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Store mouse coordinates
mouse_x = 0
mouse_y = 0

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string('''
        <h1>Webcam Feed</h1>
        <img id="webcam" src="/video_feed" style="width: 100%; max-width: 1280px;" onclick="sendMousePosition(event)" />
        <script>
            function sendMousePosition(event) {
                var x = event.clientX;
                var y = event.clientY;
                fetch('/mouse_input', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ x: x, y: y })
                });
            }
        </script>
    ''')

@app.route('/mouse_input', methods=['POST'])
def mouse_input():
    global mouse_x, mouse_y
    data = request.json
    mouse_x = data['x']
    mouse_y = data['y']
    print(f"Mouse Position: ({mouse_x}, {mouse_y})")  # You can process the mouse coordinates here
    return '', 204  # No content response

def generate_frames():
    # Set JPEG compression parameters
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    if not camera.isOpened():
        print(f"Camera with index {camera_index} not found.")
        sys.exit(1)

    app.run(host='0.0.0.0', port=5000)
