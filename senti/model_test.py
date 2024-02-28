import pytest
from sklearn.metrics import accuracy_score, classification_report
from joblib import load
from senti.sentimentanalysis import test_data, test_labels

# Load the trained model and vectorizer
classifier = load('sentiment_model.joblib')
vectorizer = load('vectorizer.joblib')

# Assuming you have a function for making predictions
def predict(text):
    features = vectorizer.transform([text])
    prediction = classifier.predict(features)[0]
    return prediction

# Test if the model can predict sentiment for positive text
def test_positive_sentiment():
    text = "I love this product!"
    assert predict(text) == 'positive'

# Test if the model can predict sentiment for negative text
def test_negative_sentiment():
    text = "This is terrible."
    assert predict(text) == 'negative'

# Test if the model can predict sentiment for neutral text
def test_neutral_sentiment():
    text = "The weather is okay"
    assert predict(text) == 'neutral'

# Test the overall accuracy of the model
def test_model_accuracy():
    # Assuming you have test_data and test_labels available
    test_features = vectorizer.transform(test_data)
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    assert accuracy > 0.8  # Set a threshold based on your expected performance

# Test the classification report
def test_classification_report():
    # Assuming you have test_data and test_labels available
    test_features = vectorizer.transform(test_data)
    predictions = classifier.predict(test_features)
    report = classification_report(test_labels, predictions)
    assert 'precision' in report  # Check if precision is present in the report
