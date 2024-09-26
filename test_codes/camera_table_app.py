from flask import Flask, render_template_string, request

class CameraTableApp:
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
            return self.render_template()

        @self.app.route('/process/<serial_number>', methods=['POST'])
        def process(serial_number):
            result = self.process_camera(serial_number)  # Call the processing function
            return self.render_template(result=result)

    def render_template(self, result=None):
        # HTML template with a table and buttons
        html_template = '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Camera Table</title>
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

    def process_camera(self, serial_number):
        # Simulated processing logic for the camera
        return f"Processed camera with serial number: {serial_number}"

    def run(self):
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app = CameraTableApp()
    app.run()
