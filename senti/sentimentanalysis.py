import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('imdb_dataset.csv')

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Display the first few rows of the dataset
print(data.head())
train_data, test_data, train_labels, test_labels = train_test_split(data['review'], data['sentiment'], test_size=0.15, random_state=42)

# Create a CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=4900)

# Fit and transform the training data
train_features = vectorizer.fit_transform(train_data)

# Transform the testing data
test_features = vectorizer.transform(test_data)

# Create a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(train_features, train_labels)

# Make predictions on the test set
predictions = classifier.predict(test_features)

# Evaluate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(test_labels, predictions))

print("saving the model")
# Save the trained model
dump(classifier, 'sentiment_model.joblib')

# Save the vectorizer
dump(vectorizer, 'vectorizer.joblib')