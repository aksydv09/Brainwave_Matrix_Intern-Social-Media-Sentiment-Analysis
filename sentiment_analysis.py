import tweepy
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Twitter API v2 credentials (Bearer Token)
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMhrwAEAAAAA2%2Bol5wISYyPdtt5AndudK3ryI7g%3DKNE4jPMweIyYhMgiuTumjIeG4s1PX3Chdg5m4BiyMBAHi3U922'  # Replace with your actual token

# Set up Tweepy with Twitter API v2 (Bearer Token Authentication)
client = tweepy.Client(bearer_token=bearer_token)

# Function to fetch tweets using Twitter API v2
def fetch_tweets(query, max_results=100):
    try:
        response = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['created_at', 'text'])
        tweets = response.data
        if tweets:
            tweet_list = [[tweet.created_at, tweet.text] for tweet in tweets]
            df = pd.DataFrame(tweet_list, columns=['Datetime', 'Text'])
            return df
        else:
            return pd.DataFrame(columns=['Datetime', 'Text'])
    except tweepy.TweepyException as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=['Datetime', 'Text'])

# Function to clean tweet text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Function to analyze sentiment of the text
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Categorize sentiment as Positive, Negative, or Neutral
def categorize_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Example usage
query = "Your Topic Here"  # Replace with the topic you want to analyze
df_tweets = fetch_tweets(query=query, max_results=100)

# Check if any tweets were returned
if df_tweets.empty:
    print("No tweets found for the given query.")
else:
    # Clean the tweets and perform sentiment analysis
    df_tweets['Cleaned_Text'] = df_tweets['Text'].apply(clean_text)
    df_tweets['Sentiment'] = df_tweets['Cleaned_Text'].apply(get_sentiment)
    df_tweets['Sentiment_Category'] = df_tweets['Sentiment'].apply(categorize_sentiment)

    # Print the first few rows of the dataframe
    print(df_tweets.head())

    # Plot sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment_Category', data=df_tweets)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.show()
