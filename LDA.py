# Author: ZHANG Si
# Object: topic classification with LDA (Latent Dirichlet Allocation)


import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer
from gensim import  models
import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')

from nltk.stem import SnowballStemmer

from langdetect import detect

# Read Data

context = pd.read_csv("D:/IRIT_stage/AUTHOT/transcrit/machine/contextMachine.csv", sep="\t", header=0,
                      names=['filename', 'filecontext'], error_bad_lines=False)
Machine = context[['filecontext']]
Machine["index"] = context.index
documents = Machine


# print(len(documents))
# print(documents[:5])

# data preprocessing : tokenization, stopwords, lemmatized

# functiopn to detect the language
def langDetect(text):
    return detect(text)

def stemmer_switch(languageabbre):
    switcher = {
        'en': 'english',
        'fr': 'french',
        'da': 'danish',
        'nl': 'dutch',
        'de': 'german',
        'hu': 'hungarian',
        'it': 'italian',
        'no': 'norwegian',
        #'en': 'porter',
        'pt': 'portuguese',
        'ro': 'romanian',
        'ru': 'russian',
        'es': 'spanish',
        'sv': 'swedish',
        'tr': 'turkish',
    }
    # print(switcher.get(languageabbre, 'invalid language'))
    return switcher.get(languageabbre, 'invalid language')


# function to define de stemmer
def stemmer_search(text):

    # englishStemmer = SnowballStemmer('english')
    # frenchStemmer = SnowballStemmer('french')
    # danishStemmer = SnowballStemmer('danish')
    # dutchStemmer = SnowballStemmer('dutch')
    # germanStemmer = SnowballStemmer('german')
    # hungarianStemmer = SnowballStemmer('hungarian')
    # italianStemmer = SnowballStemmer('italian')
    # norwegianStemmer = SnowballStemmer('norwegian')
    # porterStemmer = SnowballStemmer('porter')
    # portugueseStemmer = SnowballStemmer('portuguese')
    # romanianStemmer = SnowballStemmer('romanian')
    # russianStemmer = SnowballStemmer('russian')
    # spanishStemmer = SnowballStemmer('spanish')
    # swedishStemmer = SnowballStemmer('swedish')
    # turkishStemmer = SnowballStemmer('turkish')
    snowball_language = 'english french danish dutch german hungarian italian norwegian portuguese romanian russian ' \
                        'spanish swedish turkish '
    languageabbre = langDetect(text)
    language = stemmer_switch(languageabbre)
    if language in snowball_language:
        stemmerID = SnowballStemmer(language)
    else:
        stemmerID = None
        print('language cannot be recognised') # note: arabic language is a case as well
    return stemmerID


# function to lemmatize and stem
def lemmatize_stemming(text, stemmer):
    if stemmer is not None:
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    else:
        return None


def preprocess(text):
    result = []
    stemmer_lang = stemmer_search(text)
    for token in gensim.utils.simple_preprocess(text):   #Convert a document into a list of tokens.
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token, stemmer_lang))
    return result


if __name__ == '__main__':
    __spec__ = None

    # example to preview
    doc_sample = documents[documents['index'] == 310].values[0][0]
    doc_sample = documents[documents['index'] == 6].values[0][0]
    print('original document: ')
    words = []
    for word in doc_sample.split(' '):
        words.append(word)
    print(words)
    print('\n\n tokenized and lemmatized document: ')
    print(preprocess(doc_sample))


    doc_sample = documents[documents['index'] == 7].values[0][0]
    print('original document: ')
    words = []
    for word in doc_sample.split(' '):
        words.append(word)
    print(words)
    print('\n\n tokenized and lemmatized document: ')
    print(preprocess(doc_sample))


    #### Preprocess the headline text, saving the results as ‘processed_docs’

    processed_docs = documents['filecontext'].map(preprocess)
    processed_docs[5:10]

    print(len(processed_docs))
    print(processed_docs[5:10])

    # Bag of words on the data set

    dictionary = gensim.corpora.Dictionary(processed_docs)

    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 100:
            break

    ### filter out tokens that appear in
    # less than 15 documents(absolute number) or
    # more than 0.5 documents (fraction of total corpus size, not absolute number).
    # after the above two steps, keep only the first 100000 most frequent tokens.

    dictionary.filter_extremes(no_below=15, no_above=0.3, keep_n=100000)

    # create a dictionary reporting how many  words and how many times those words appear. Save this to ‘bow_corpus’

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    print('bag words: \n', bow_corpus[300:310])
    bow_doc_310 = bow_corpus[310]

    for i in range(len(bow_doc_310)):
        print(
            "Word {} (\"{}\") appears {} time.".format(bow_doc_310[i][0], dictionary[bow_doc_310[i][0]], bow_doc_310[i][1]))


    ### TF-IDF
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    from pprint import pprint

    for doc in corpus_tfidf:
        pprint(doc)
        break

    ### LDA using Bag of words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=15, id2word=dictionary, passes=2, workers=3)

    for idx, topic in lda_model.print_topics(-1, num_words=20):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=15, id2word=dictionary, passes=2, workers=3)

    for idx, topic in lda_model_tfidf.print_topics(-1, num_words=20):
        print('Topic: {} Word: {}'.format(idx, topic))

    ## Performance evaluation by classifying sample document using LDA Bag of Words model

    processed_docs[310]

    for index, score in sorted(lda_model[bow_corpus[310]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

    ## Performance evaluation by classifying sample document using LDA TF-IDF model.

    for index, score in sorted(lda_model_tfidf[bow_corpus[310]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

    ## Testing model on unseen document

    unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))

    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))