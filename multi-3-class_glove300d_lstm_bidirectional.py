#Importing libraries

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,GRU,CuDNNGRU,RNN,CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,Input, SpatialDropout1D, add, concatenate,GlobalMaxPool1D
from keras.layers.embeddings import Embedding


from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

## Plot
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import matplotlib as plt

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

#Converting into pandas dataframe and filtering only text and ratings given by the users
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',10000)
data_frame_yelp_business=pd.DataFrame()
print('Data is loading!')
filename='yelp-dataset/cs_data/review.csv'
i=0
for chunk in pd.read_csv(filename,
                          chunksize=10000):
    data_frame_yelp_business=data_frame_yelp_business.append(chunk,ignore_index=True)
    i+=1
    if i>=2:
        break
    print('.',end='')
print('Data is loaded')
data_frame_yelp_business.drop(['Unnamed: 0'],axis=1,inplace=True)
df=pd.DataFrame(data_frame_yelp_business)



df= df.dropna()
df=df[['stars','text']]
df = df[df.stars.apply(lambda x: x !="")]
df = df[df.text.apply(lambda x: x !="")]
print(df.head())

print(df.describe())


print(df['stars'].unique())
label = df['stars'].map(lambda x : 2 if int(x) > 3 else ( 1 if int(x) == 3 else 0))
labels = to_categorical(label, num_classes=3)

unique, counts = np.unique(labels, return_counts=True)

print (np.asarray((unique, counts)).T)
def clean_text(text):

    text = text.translate(string.punctuation)

    text = text.lower().split()

    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


df['text'] = df['text'].map(lambda x: clean_text(x))

print(df.head(10))


vocabulary_size = 60000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=300)
print(data.shape)

#gloove
embeddings_index = dict()
f = open('Glove_files/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


from keras.optimizers import Adam


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector



model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 300, input_length=300, weights=[embedding_matrix], trainable=False))
model_glove.add(LSTM(100))
model_glove.add(Dense(3, activation='softmax'))
model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



X=data
labels=np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)


model_glove.fit(X_train,y_train , validation_split=0.2, epochs = 3,batch_size=52
                ,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])


accr = model_glove.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))




import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


