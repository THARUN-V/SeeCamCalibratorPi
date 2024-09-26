from flask import Flask, render_template_string, request

from CamContext import *

class CameraCalibrationApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.cam_context = CamContext()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return self.render_template(result=None)

        @self.app.route('/calibrate', methods=['POST'])
        def calibrate():
            result = self.camera_calibration()  # Call the calibration function
            return self.render_template(result=result)

    def render_template(self, result):
        # HTML template with a heading and a button
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
            <form method="POST" action="/calibrate">
                <button type="submit">Start Calibration</button>
            </form>
            {% if result %}
                <h2>Result: {{ result }}</h2>
            {% endif %}
        </body>
        </html>
        '''
        return render_template_string(html_template, result=result)

    def camera_calibration(self):
        # Here you would add your actual calibration logic
        # For now, we simulate with a simple message
        # return "Camera calibration completed successfully!"
        see_cams = self.cam_context.get_seecam()
        
        if see_cams == None:
            return "!!!! No cameras Found !!!!"
        else:
            for cam in see_cams:
                print(cam.serial_number)
                print(cam.camera_index)
            return f"{len(see_cams)} Cameras Found"
        
        

    def run(self):
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app = CameraCalibrationApp()
    app.run()
