from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)

@app.route("/")
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "message" not in data:
            return jsonify({"error": "No 'message' key found in JSON data"}), 400

        message = data["message"]
        if not isinstance(message, str):
            return jsonify({"error": "'message' must be a string"}), 400

        response = get_response(message)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

