import cv2
import sys
from flask import Flask, Response, request, jsonify

# Camera class to handle video streaming
class Camera:
    def __init__(self, index=0):
        # Initialize camera with the provided index
        self.camera = cv2.VideoCapture(index)
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # JPEG compression quality (0-100)

        if not self.camera.isOpened():
            raise ValueError(f"Camera with index {index} not found.")

    def generate_frames(self):
        # Read and encode frames
        while True:
            success, frame = self.camera.read()  # Read frame from camera
            if not success:
                break
            else:
                # Encode the frame in JPEG format with specified quality
                ret, buffer = cv2.imencode('.jpg', frame, self.encode_param)
                frame = buffer.tobytes()

                # Yield the frame as a byte stream for the response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def __del__(self):
        # Release the camera resource when done
        if self.camera.isOpened():
            self.camera.release()

# WebcamApp class to handle the Flask app and routes
class WebcamApp:
    def __init__(self, camera_index=0):
        self.app = Flask(__name__)
        self.camera = Camera(camera_index)

        # Define routes
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/mouse_click', 'mouse_click', self.mouse_click, methods=['POST'])

    def video_feed(self):
        # Route to stream the video feed
        return Response(self.camera.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def index(self):
        # Basic HTML page to display the video feed and capture mouse clicks
        return '''
            <h1>Webcam Feed</h1>
            <img id="webcam_image" src="/video_feed" />

            <h2>Click on the image to get the coordinates</h2>
            <p>Click anywhere on the image to capture the mouse coordinates.</p>

            <script>
                // JavaScript to handle mouse clicks and get coordinates
                document.getElementById("webcam_image").addEventListener("click", function(event) {
                    // Get the x and y coordinates relative to the image
                    var rect = event.target.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;

                    // Send the coordinates to the Flask backend via AJAX (fetch)
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
        # Handle mouse click input
        data = request.json
        x = data.get('x')
        y = data.get('y')

        # You can now use the (x, y) coordinates for any purpose
        print(f"Mouse clicked at: ({x}, {y})")

        # Respond back to the client with a confirmation
        return jsonify({"status": "success", "x": x, "y": y})

    def run(self, host='0.0.0.0', port=5000):
        # Run the Flask app
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    # Get camera index from command line arguments or default to 0
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Create an instance of the webcam app and run it
    app = WebcamApp(camera_index)
    app.run()
