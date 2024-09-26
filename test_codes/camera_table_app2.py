from flask import Flask, render_template_string, request

class CameraCalibrationApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.cameras = {
            "12345": "Camera Device 1",
            "67890": "Camera Device 2",
            "24680": "Camera Device 3"
        }  # Example camera dictionary

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return self.render_start_page()

        @self.app.route('/start_calibration', methods=['POST'])
        def start_calibration():
            return self.render_camera_table()

        @self.app.route('/process/<serial_number>', methods=['POST'])
        def process(serial_number):
            device_name = self.cameras.get(serial_number)
            result = f"Camera Serial Number: {serial_number}, Device Name: {device_name}"
            return self.render_camera_table(result=result)

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

    def render_camera_table(self, result=None):
        # HTML template with a table and buttons
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
                </tr>
                {% for serial, device in cameras.items() %}
                <tr>
                    <td>{{ serial }}</td>
                    <td>{{ device }}</td>
                    <td>
                        <form method="POST" action="/process/{{ serial }}">
                            <button type="submit">Process</button>
                        </form>
                    </td>
                </tr>
                {% endfor %}
            </table>
            {% if result %}
                <h2>Result: {{ result }}</h2>
            {% endif %}
        </body>
        </html>
        '''
        return render_template_string(html_template, cameras=self.cameras, result=result)

    def run(self):
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app = CameraCalibrationApp()
    app.run()
