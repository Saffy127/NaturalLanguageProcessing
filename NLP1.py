import nltk
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    if sentiment_polarity > 0:
        sentiment = 'positive'
    elif sentiment_polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment, sentiment_polarity, sentiment_subjectivity

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    text = "This is an amazing product. I am extremely happy with it."
    sentiment, polarity, subjectivity = analyze_sentiment(text)

    print(f"Sentiment: {sentiment}")
    print(f"Polarity: {polarity}")
    print(f"Subjectivity: {subjectivity}")
