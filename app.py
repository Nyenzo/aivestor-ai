from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Aivestor AI API'

if __name__ == '__main__':
    app.run(port=5001, debug=True)