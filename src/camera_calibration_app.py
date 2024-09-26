import sys
import cv2
from flask import Flask, render_template_string, Response

from CamContext import *

class CameraCalibrationApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.cam_context = CamContext()
        
        self.cameras = self.get_cams()  # Get available cameras
        self.setup_routes()  # Setup routes for the app

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return self.render_start_page()

        @self.app.route('/start_calibration', methods=['POST'])
        def start_calibration():
            return self.render_camera_table()

        @self.app.route('/video_feed/<serial_number>')
        def video_feed(serial_number):
            if serial_number in self.cameras:
                return Response(self.generate_frames(serial_number), mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                print(f"Camera with serial number {serial_number} not found.")  # Debug log
                return "Camera not found", 404

    def generate_frames(self, serial_number):
        device_index = self.cameras[serial_number]
        cap = cv2.VideoCapture(device_index)

        # Set camera resolution to reduce load (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            success, frame = cap.read()  # Read a frame from the camera
            if not success:
                break
            
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()  # Convert to byte array
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()  # Release the camera when done

    def render_start_page(self):
        # HTML template for the start page
        html_template = '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Camera Calibration</title>
        </head>
        <body>
            <h1>Camera Calibration</h1>
            <form method="POST" action="/start_calibration">
                <button type="submit">Start Calibration</button>
            </form>
        </body>
        </html>
        '''
        return render_template_string(html_template)

    def render_camera_table(self):
        html_template = '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Camera List</title>
            <style>
                table { width: 50%; margin: 20px auto; border-collapse: collapse; }
                th, td { padding: 10px; border: 1px solid #ddd; text-align: center; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Camera List</h1>
            <table>
                <tr>
                    <th>Serial Number</th>
                    <th>Device Name</th>
                    <th>Action</th>
                    <th>Preview</th>
                </tr>
                {% for serial, device in cameras.items() %}
                <tr>
                    <td>{{ serial }}</td>
                    <td>{{ device }}</td>
                    <td>
                        <button onclick="startCalibration('{{ serial }}')">Start Calibration</button>
                    </td>
                    <td>
                        <img id="camera-image-{{ serial }}" src="" alt="No Image" style="width: 200px; height: auto;">
                    </td>
                </tr>
                {% endfor %}
            </table>
            <script>
                function startCalibration(serial) {
                    const imgElement = document.getElementById('camera-image-' + serial);
                    imgElement.src = '/video_feed/' + serial;  // Start video feed
                }
            </script>
        </body>
        </html>
        '''
        return render_template_string(html_template, cameras=self.cameras)

    def get_cams(self):
        see_cams = self.cam_context.get_seecam()
        
        if see_cams is None:
            print("!!!!!!!!! No Cameras Connected !!!!!!!!!!")
            sys.exit()
        else:
            camera_dict = {cam.serial_number: cam.camera_index for cam in see_cams}
            print("Connected Cameras:", camera_dict)  # Log the available cameras
            return camera_dict

    def run(self):
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app = CameraCalibrationApp()
    app.run()
