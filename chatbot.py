# Chatbot using NLP and Neural Networks
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# nltk.download("punkt")
# nltk.download("wordnet")
# print("after download")

data = {"intents": [
    {
        "tag": "greeting",
        "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up"],
        "responses": ["Howdy partner", "Hello", "How are you doing?", "Greetings!", "How do you do?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you?", "When were you born?", "When is your birthday?"],
        "responses": ["I am 21 years old", "I was born in 2000", "My birthday is 17-01-2000", "17-01-2000"]
    },
    {
        "tag": "date",
        "patterns": ["What are you doing this weekend?", "Do you want to hangout sometime?", "What are your plans for "
                                                                                             "this week?"],
        "responses": ["I am available all week", "I don't have ay plas", "I am not busy"]
    },
    {
        "tag": "name",
        "patterns": ["Whats your name", "What are you called?", "Who are you?"],
        "responses": ["My name is Grogu", "I'm Grogu", "Grogu"]
    },
    {
        "tag": "goodbye",
        "patterns": ["bye", "g2g", "see ya", "adios", "cya"],
        "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
    },
]}

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# each list to create
words = []
classes = []
doc_x = []
doc_y = []

# loop through all the intents
# tokenize each pattern and append tokens to words
# append patterns and the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])

    # add the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

print(words)
print(len(words))
print(doc_x)
print(doc_y)
print(classes)

# lemmatize all the words in the vocab and convert them to lowercase if the words don't appear in punctuation
# string.punctuation has all the punctuations : !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplication occur
words = sorted(set(words))
classes = sorted(set(classes))

print("words:", words)
print(len(words))
print("classes:", classes)

# Now that we have separated our data, we are now ready to train our algorithm
# However, Neural Network expect numerical values, and not words
# In order to convert our data to numerical values, we are going to leverage a technique called bag of words

training = []
out_empty = [0] * len(classes)
print("out_empty:", out_empty)

# Creating the bag of words model
print(enumerate(doc_x))
for idx, doc in enumerate(doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    print(text)

