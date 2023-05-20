from flask import Flask, render_template, request
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_analysis_spanish.sentiment_analysis import SentimentAnalysisSpanish

app = Flask(__name__)


@Language.factory('language_detector')
def get_lang_detector(nlp, name):
    return LanguageDetector()


def get_ents(rawtext, taskoption):
    # python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('language_detector', last=True)
    doc = nlp(rawtext)
    language = doc._.language['language']

    if language != 'en':
        # python -m spacy download es_core_news_sm
        nlp = spacy.load('es_core_news_sm')
        doc = nlp(rawtext)

    return [ent.text for ent in doc.ents if ent.label_ == taskoption], language


def get_sentiment_analysis(rawtext, language):
    if language == 'en':
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()

        return sid.polarity_scores(rawtext)['compound']
    else:
        sas = SentimentAnalysisSpanish()

        return sas.sentiment(rawtext)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    rawtext = request.form['rawtext']
    taskoption = request.form['taskoption']

    results, language = get_ents(rawtext, taskoption)
    num_of_results = len(results)
    sentiment_analysis = get_sentiment_analysis(rawtext, language)

    return render_template('index.html', results=results, num_of_results=num_of_results, sentiment_analysis=sentiment_analysis, language=language)
