from flask import Flask, render_template, request
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

app = Flask(__name__)


@Language.factory('language_detector')
def get_lang_detector(nlp, name):
    return LanguageDetector()


def get_ents(rawtext, taskoption):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('language_detector', last=True)
    doc = nlp(rawtext)

    if doc._.language['language'] == 'es':
        nlp = spacy.load('es_core_news_sm')
        doc = nlp(rawtext)

    return [ent.text for ent in doc.ents if ent.label_ == taskoption]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    rawtext = request.form['rawtext']
    taskoption = request.form['taskoption']

    results = get_ents(rawtext, taskoption)
    num_of_results = len(results)

    return render_template('index.html', results=results, num_of_results=num_of_results)
