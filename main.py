import pandas as pd

import nltk,re
from gensim.models import Word2Vec
#import gensim
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

#nltk.download('stopwords')

#W2V_SIZE = 300
#W2V_WINDOW = 7
#W2V_EPOCH = 32
#W2V_MIN_COUNT = 10

train_df= pd.read_csv("/home/rishabh/NLP/hate speech/data.1/train_E6oV3lV.csv",index_col="id")
test_df=pd.read_csv("/home/rishabh/NLP/hate speech/data.1/test_tweets_anuFYb8.csv",index_col="id",)

#train_df= train_df.iloc[1900:2000,:]

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

stop_words=stopwords.words("english")
stemmer = SnowballStemmer("english",ignore_stopwords=True)

def preprocess(text,stem= True):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)

    return " ".join(tokens)

#print(preprocess("love loving ,tvx lovely"))
train_df['tweet1'] = train_df['tweet'].apply(lambda text: preprocess(text))


documents = [t1.split() for t1 in train_df.tweet1] 

#w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,window=W2V_WINDOW,min_count=W2V_MIN_COUNT, workers=8)
#w2v_model.build_vocab(documents)
#
#
#words2 = w2v_model.wv.vocab.keys()
#vocab_size = len(words2)
#print("Vocab size", vocab_size)
#
#w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
#w2v_model.most_similar("hate")

model = Word2Vec(documents)


words = model.wv.vocab

## Finding Word Vectors
#vector = model.wv['love']

# Most similar words
similar = model.wv.most_similar('hate')


