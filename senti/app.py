from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the trained model and vectorizer
classifier = load('sentiment_model.joblib')
vectorizer = load('vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        features = vectorizer.transform([text])
        prediction = classifier.predict(features)[0]
        return render_template('result.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
