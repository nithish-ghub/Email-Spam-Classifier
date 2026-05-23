import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        message_vector = vectorizer.transform([message])
        result = model.predict(message_vector)

        if result[0] == 1:
            prediction = "SPAM"
        else:
            prediction = "HAM"

    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)