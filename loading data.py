# Import necessary libraries
import pandas as pd  
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load Fake News dataset
df_fake = pd.read_csv("Fake.csv")
df_fake["label"] = "FAKE"  # Add label column

# Load Real News dataset
df_real = pd.read_csv("True.csv")
df_real["label"] = "REAL"  # Add label column

# Combine both datasets
df = pd.concat([df_fake, df_real], ignore_index=True)

# Shuffle the dataset to avoid order bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display dataset overview
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print(df.info())  # Check data types and missing values
print("\nMissing values per column:\n", df.isnull().sum())  # Count missing values

# Drop unnecessary columns (subject and date)
df.drop(columns=['subject', 'date'], inplace=True)

# Ensure all necessary columns exist
assert 'title' in df.columns, "Error: 'title' column is missing!"
assert 'text' in df.columns, "Error: 'text' column is missing!"
assert 'label' in df.columns, "Error: 'label' column is missing!"

# Convert labels to numerical format (FAKE = 0, REAL = 1)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Verify label conversion
print("\nLabel Data Type:", df['label'].dtype)  # Should print 'int64'
print("Unique Labels:", df['label'].unique())  # Should print [0, 1]

# Ensure label column is correctly mapped
assert df['label'].isin([0, 1]).all(), "Error: Label mapping failed!"

# Download necessary NLTK packages (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text (removing special characters, numbers, extra spaces)
def clean_text(text):
    if not isinstance(text, str):  # Ensure input is a string
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(words)

# Apply cleaning function to 'text' column
df['text'] = df['text'].apply(clean_text)

# Ensure no missing or empty text values
df['text'] = df['text'].astype(str).fillna("")
print("\nMissing values in 'text':", df['text'].isnull().sum())  # Should be 0

# Check text length distribution
print("\nText Length Statistics:\n", df['text'].apply(lambda x: len(str(x))).describe())

# Ensure the dataset is not empty
assert len(df) > 0, "Error: Dataset is empty after processing!"

# Final dataset preview
print("\nFinal Dataset Sample:\n", df.head())

# Save the cleaned dataset
df.to_csv("cleaned_news_dataset.csv", index=False)
print("\nPreprocessed dataset saved as 'cleaned_news_dataset.csv'!")




