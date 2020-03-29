import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import python_service_pb2


def generate_frequency_distribution(id, content, total):
    stop_words = stopwords.words('portuguese')
    tokens = nltk.word_tokenize(content, language='portuguese')
    tokens_filtered_stopwords = []
    for w in tokens:
        if w.lower() not in stop_words:
            tokens_filtered_stopwords.append(w)
    # # remove all the not is alfabethic
    tokens_filtered_pontuaction = []
    for w in tokens_filtered_stopwords:
        if w.isalpha():
            tokens_filtered_pontuaction.append(w)
    # Frequency Distribution
    fdist = FreqDist(tokens_filtered_pontuaction)
    most_commons = fdist.most_common(total)
    result = python_service_pb2.FrequencyDistribution()
    result.idDocument = id
    for tupla in most_commons:
        frequency = python_service_pb2.Frequency()
        frequency.word = tupla[0]
        frequency.quantity = tupla[1]
        result.frequencies.append(frequency)
    return result







