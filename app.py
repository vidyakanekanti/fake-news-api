from flask import Flask, request, jsonify
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Use absolute paths for loading the model and vectorizer
BASE_DIR = r"C:\Users\sruja\Desktop\EDUCATION\vidya project"
model_path = os.path.join(BASE_DIR, "news_classifier_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)  # Stop execution if files are missing

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get input data (JSON)
        text = data['text']  # Extract the 'text' field

        # Transform the text using the vectorizer
        text_tfidf = vectorizer.transform([text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        label = "Fake News" if prediction == 0 else "Real News"

        return jsonify({"prediction": label})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
