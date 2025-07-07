# Task 3: NLP with spaCy - NER + Rule-based Sentiment Analysis

import spacy
from textblob import TextBlob

# Step 1: Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")  # Make sure to run: python -m spacy download en_core_web_sm

# Sample Amazon reviews
reviews = [
    "I love my new Apple iPhone 13. It's fast and the camera is amazing!",
    "This Samsung TV stopped working after a month. Very disappointed.",
    "The Nike Air Max shoes are so comfortable and stylish!",
    "Avoid buying this laptop. Dell should be ashamed of this poor quality.",
    "I’m happy with my Sony headphones. Great value for the price!"
]

# Step 2: Loop through each review and apply NER + sentiment analysis
for i, review in enumerate(reviews):
    print(f"\nReview {i+1}: {review}")
    
    # Named Entity Recognition
    doc = nlp(review)
    print("Named Entities:")
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            print(f" - {ent.text} ({ent.label_})")

    # Sentiment Analysis using TextBlob (rule-based)
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    print(f"Sentiment: {sentiment} (Polarity Score: {polarity:.2f})")
