from flask import Flask, render_template_string, request

app = Flask(__name__)

# Define the function that will be executed when the button is pressed
def some_function():
    print("Button was pressed!")

# HTML template with a button
html_template = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Button Example</title>
</head>
<body>
    <h1>Press the Button</h1>
    <form method="POST" action="/button">
        <button type="submit">Press Me!</button>
    </form>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/button', methods=['POST'])
def button_pressed():
    some_function()  # Call the function when the button is pressed
    return "Button pressed! Check the console.", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
