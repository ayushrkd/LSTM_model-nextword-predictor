# import tensorflow as tf
import keras
import pickle
import streamlit as st
import time
import numpy as np


# Load your Keras model
model = keras.models.load_model('lstm_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



def prep_input(text):

    # text = "where are"

    for i in range(3):
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = keras.preprocessing.sequence.pad_sequences([token_text], maxlen=12, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                # print(text)
                print(text, end=" ")
                time.sleep(2)

    # print(text)

st.title("hello world ")