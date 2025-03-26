import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed dataset
df = pd.read_csv("cleaned_news_dataset.csv")

# Check dataset structure
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print(df.info())

# Ensure 'text' column is of type string
df['text'] = df['text'].astype(str)

# Split data into features (X) and labels (y)
X = df['text']
y = df['label']

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features to 5000 for performance
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model and vectorizer for future use
import joblib
joblib.dump(classifier, "news_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
