import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#module setups and download
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

#fucntion for cleaning the data to remove noise
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stopwords]
    return ' '.join(text)

#loading the dataset
df = pd.read_csv("IMDB Dataset.csv")
#adding a new column cleaned_review to store cleaned data
df['cleaned_review'] = df['review'].apply(clean_text)

#calculating the positive and negative words
positive_words = ' '.join(df[df['sentiment'] == 'positive']['cleaned_review']).split()
negative_words = ' '.join(df[df['sentiment'] == 'negative']['cleaned_review']).split()
#selecting  top 20 positive and negative words
pos_freq = Counter(positive_words).most_common(20)
neg_freq = Counter(negative_words).most_common(20)
#converting the words to a dataframe
pos_df = pd.DataFrame(pos_freq, columns=['Word', 'Frequency'])
neg_df = pd.DataFrame(neg_freq, columns=['Word', 'Frequency'])

#saving the cleaned dataframe to csv
df.to_csv("cleaned_imdb_reviews.csv", index=False)

if __name__ == '__main__':
    #plotting these in the graph
    fig, axes = plt.subplots(1,2, figsize=(15,5))
    sns.barplot(x='Frequency', y='Word', data=pos_df, ax=axes[0])
    sns.barplot(x='Frequency', y='Word', data=neg_df, ax=axes[1])
    axes[0].set_title('Top 20 Positive Words')
    axes[1].set_title('Top 20 Negative Words')
    plt.show()
    #displaying the total number of counts for positive and negative
    sns.countplot(x='sentiment', data=df)
    plt.title('sentiment distribution')
    plt.show()
