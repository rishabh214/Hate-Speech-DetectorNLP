
import pandas as pd 
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]


model = keras.models.load_model('/home/rishabh/NLP/hate speech/model.h5')
df = pd.read_csv('/home/rishabh/NLP/hate speech/data1.1.csv', encoding ="ISO-8859-1")
df.dropna(subset=['text'], inplace=True)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

def decode_sentiment(score):
    return 'HATE' if score < 0.5 else 'NON HATE'


def predict(text):
    
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=300)
    score = model.predict([x_test])[0]
    label = decode_sentiment(score)

    return {"label": label, "score": float(score)}
    

while(True):
    str1=input("Enter text\n")
    print(predict(str1))
    print()

    print("exit? (q)")
    i=input()
    if (i=='q'):
        break
    
