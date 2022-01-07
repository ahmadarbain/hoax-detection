import nltk
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import keras.backend as K

nltk.download('stopwords')
nltk.download('punkt')

from keras import models
from nltk.data import load
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from sklearn.metrics import accuracy_score

def marks_prediction(text):

    slang = pd.read_csv("colloquial-indonesian-lexicon.csv", error_bad_lines=False)
    slang_dict ={}
    for i in range(len(slang)):
        slang_dict[slang.iloc[i]['slang']] =  slang.iloc[i]['formal']

    dataset = pd.read_csv("dataset-berita-indonesia.csv", sep=';')
    tokenizer = Tokenizer(num_words=2000, split=' ')
    tokenizer.fit_on_texts(dataset['berita'].values)

    def remove_numbers(text):
        result=re.sub(r'\d+', '', text)
        return result

    def remove_whitespace(text):
        return " ".join(text.split())

    def deEmojify(text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF" 
            u"\U0001F1E0-\U0001F1FF"  
                            "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)

    def _normalize_contractions_text(text):
        contractions = slang_dict
        new_token_list = []
        token_list = text.split()
        for word_pos in range(len(token_list)):
            word = token_list[word_pos]
            first_upper = False
            if word[0].isupper():
                first_upper = True
            if word.lower() in contractions:
                replacement = contractions[word.lower()]
                if first_upper:
                    replacement = replacement[0].upper()+replacement[1:]
                replacement_tokens = replacement.split()
                if len(replacement_tokens)>1:
                    new_token_list.append(replacement_tokens[0])
                    new_token_list.append(replacement_tokens[1])
                else:
                    new_token_list.append(replacement_tokens[0])
            else:
                new_token_list.append(word)

        sentence = " ".join(new_token_list).strip(" ")
        return(sentence)


    def lemmatize(kalimat):
        tokens = word_tokenize(kalimat)
        listStopword =  set(stopwords.words('indonesian'))
        
        removed = []
        for t in tokens:
            if t not in listStopword:
                removed.append(t)
        cleaned = " ".join(removed)
        return cleaned

    def clean(kalimat):
        cleaned = remove_numbers(kalimat)
        cleaned = remove_whitespace(cleaned)
        cleaned = deEmojify(cleaned)
        cleaned = _normalize_contractions_text(cleaned)
        cleaned = lemmatize(cleaned)
        return cleaned

    sentences_raw = dataset['berita']
    sentences = sentences_raw.apply(clean)

    labels = dataset['kategori']
    labels = labels.apply(lambda x: 1 if x == 'hoax' else 0)

    X = sentences
    y = labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify = y, random_state=42)

    all_train_sentences = X.values.tolist()

    train_sentences = X_train.values.tolist()
    train_labels = y_train.values.tolist()

    val_sentences = X_val.values.tolist()
    val_labels = y_val.values.tolist()

    #vocab_size = 100000
    embedding_dim = 300
    max_length = 750
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"


    #initialize tokenizer based on all training sentences(before split)
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(all_train_sentences)

    #make indexing for each word
    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    #tokenize with padding X_train
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, truncating=trunc_type)
    train_padded = np.asarray(train_padded)
    train_labels = np.asarray(train_labels)

    #tokenize with padding X_val
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded = pad_sequences(val_sequences, padding=padding_type, truncating=trunc_type)
    val_padded = np.asarray(val_padded)
    val_labels = np.asarray(val_labels)

    def clean_token_pad(text):
        text_list = []
        text_list.append(text)
        sequences = tokenizer.texts_to_sequences(text_list)
        padded = pad_sequences(sequences, padding=padding_type, truncating=trunc_type)
        padded = np.asarray(padded)
        return padded

    
    def f1(y_pred, y_true): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    
 
    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.AUC(name='auc'),
        f1
    ]

    # model_multi = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vocab_size+1, embedding_dim),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    # model_multi.compile(loss='binary_crossentropy',optimizer='adam',metrics=METRICS)
    # model_multi.summary()

    # num_epochs = 3
    # model_multi.fit(train_padded, train_labels, validation_split = 0.1, epochs=num_epochs, verbose=2)
    # print("Training Complete")

    load_Model = tf.keras.models.load_model("model.h5", custom_objects = {'f1' : f1})
    text_token=np.asarray(clean_token_pad(text))
    prediction = load_Model.predict(text_token)[0]
    hoax_detection = int(load_Model.predict(text_token).round())
    if hoax_detection == 0:
        result = "Hoax"
    elif hoax_detection == 1:
        result = "Fakta"

    # text=clean_token_pad(text)
    # print(model_multi.predict(text)[0])
    # print(int(model_multi.predict(text).round()))

    # return model_multi
    return [
        prediction[0],
        result
    ]

  


