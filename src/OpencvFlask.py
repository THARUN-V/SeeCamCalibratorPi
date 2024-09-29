import cv2
import sys
import threading
from flask import Flask, Response, request, jsonify

# Camera class to handle video streaming
class Camera:
    def __init__(self, calib_obj):
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # JPEG compression quality (0-100)
        self._calib_obj = calib_obj
    
    def generate_frames(self):
        while True:
            if self._calib_obj.queue_display.qsize() > 0:
                frame = self._calib_obj.queue_display.get()
                                
                ret, buffer = cv2.imencode('.jpg', frame, self.encode_param)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
    def __del__(self):
        pass

# WebcamApp class to handle Flask app and routes
class WebcamApp:
    def __init__(self, camera_index=0):
        self.app = Flask(__name__)
        self.camera = Camera(camera_index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/mouse_click', 'mouse_click', self.mouse_click, methods=['POST'])
        self.app.add_url_rule('/shutdown', 'shutdown', self.shutdown)  # Add route to shutdown the server

    def video_feed(self):
        return Response(self.camera.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def index(self):
        return '''
            <h1>Webcam Feed</h1>
            <img id="webcam_image" src="/video_feed" />
            <h2>Click on the image to get the coordinates</h2>
            <p>Click anywhere on the image to capture the mouse coordinates.</p>
            <script>
                document.getElementById("webcam_image").addEventListener("click", function(event) {
                    var rect = event.target.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;
                    fetch("/mouse_click", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            x: x,
                            y: y
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log("Mouse coordinates sent to the backend:", data);
                    })
                    .catch(error => console.error("Error:", error));
                });
            </script>
        '''

    def mouse_click(self):
        data = request.json
        x = data.get('x')
        y = data.get('y')
        print(f"Mouse clicked at: ({x}, {y})")
        return jsonify({"status": "success", "x": x, "y": y})

    def shutdown(self):
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return 'Server shutting down...'

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)

# Function to run the Flask app in a separate thread
def start_webcam_app(camera_index):
    app = WebcamApp(camera_index)
    thread = threading.Thread(target=app.run)
    thread.start()
    return app, thread

# Function to stop the Flask app
def stop_webcam_app(app):
    # Send request to /shutdown to stop Flask server
    import requests
    try:
        requests.get('http://127.0.0.1:5000/shutdown')
    except Exception as e:
        print(f"Error stopping app: {e}")

if __name__ == '__main__':
    # Get camera index from command line arguments or default to 0
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Start the Flask app in a separate thread, passing the camera_index
    app, thread = start_webcam_app(camera_index)

    try:
        while thread.is_alive():
            thread.join(1)  # Keep the main thread alive while Flask runs
    except KeyboardInterrupt:
        print("Shutting down server...")
        stop_webcam_app(app)
        thread.join()
