import pandas as pd

import nltk,re
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

#nltk.download('stopwords')

  
train_df= pd.read_csv("/home/rishabh/NLP/hate speech/data.1/train_E6oV3lV.csv",index_col="id")
test_df=pd.read_csv("/home/rishabh/NLP/hate speech/data.1/test_tweets_anuFYb8.csv",index_col="id")

#train_df= train_df.iloc[1900:2000,:]
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

stop_words=stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text,):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(token)
    return " ".join(tokens)


train_df['tweet1'] = train_df['tweet'].apply(lambda x: preprocess(x))
