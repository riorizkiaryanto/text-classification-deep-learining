from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from predict import prediction

_CLIENT_URL = "http://127.0.0.1:4200"

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": _CLIENT_URL}})

@app.route('/')
def index():
    return render_template('../client/index.html')

@app.route('/predict', methods=['POST'])
def classify_text():
    data = request.get_json()
    text = data['text']

    prediction_process = prediction(text)
    response = prediction_process.run()

    print(text)
    print(response)

    return jsonify(response)