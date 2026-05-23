import joblib

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("=== Email Spam Classifier ===")

while True:
    # Get user input
    message = input("\nEnter a message: ")

    # Convert text into vector
    message_vector = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_vector)

    # Output
    if prediction[0] == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM")

    # Continue or stop
    choice = input("\nDo you want to test another message? (yes/no): ")

    if choice.lower() != "yes":
        print("Program ended.")
        break