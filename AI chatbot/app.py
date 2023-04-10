from flask import Flask, request, jsonify, make_response
import random
import json
import pickle
import threading
import numpy as np
from flask_caching import Cache
from flask_cors import CORS
import nltk

from nltk.stem import WordNetLemmatizer

from keras.models import load_model
app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@cache.cached(timeout=300)
@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    response_event = ThreadSafeVariable()
    t = threading.Thread(target=handle_chat_request, args=(message, response_event,))
    t.start()
    print(f"Thread started for message: {message}")
    response_event.wait()
    response = {'message': response_event.data}
    return make_response(jsonify(response), 200)


def handle_chat_request(message, response_event):
    ints = predict_class(message)
    res = get_response(ints, intents)
    response_event.data = res
    response_event.set(res)


class ThreadSafeVariable:
    def __init__(self):
        self._value = None
        self._event = threading.Event()

    def set(self, value):
        self._value = value
        self._event.set()

    def wait(self):
        self._event.wait()

    def get(self):
        return self._value


if __name__ == '__main__':
    print(" Welcome to Pollmonk Equiry Chatbot System!")

    app.run(debug=True)
