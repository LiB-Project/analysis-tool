import logging
from collections import OrderedDict

from pre_process import *
import os
import gensim
import yaml
import pickle

from util import Util

logger = logging.getLogger(__name__)

# Load configuration
with open('config.yml') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)
fp.close()


class LdaModelGenerator:
    def __init__(self):
        if not os.path.exists('data'):
            os.makedirs('data')

    def execute(self, id_doc, conteudo):
        processor = TextPreProcessor('portuguese')
        new_doc_preprocessed = processor.execute_pre_process(conteudo)

        if Util.docs_preprocessed_exists():
            docs_preprocessed = Util.read_docs_preprocessed()
            docs_preprocessed[id_doc] = new_doc_preprocessed
            Util.save_docs_preprocessed(docs_preprocessed)
        else:
            docs_preprocessed = OrderedDict({id_doc: new_doc_preprocessed})
            Util.save_docs_preprocessed(docs_preprocessed)

        docs_preprocessed_list = Util.read_docs_preprocessed()
        dictionary = gensim.corpora.Dictionary(docs_preprocessed_list.values())
        dictionary.filter_extremes(no_below=1, no_above=0.7)
        dictionary.save(config['paths']['dictionary'])

        docs_bow = {}
        for k, v in docs_preprocessed_list.items():
            if len(v) > 0:
                docs_bow[k] = dictionary.doc2bow(v)
            else:
                docs_bow[k] = []

        Util.save_docs_bow(docs_bow)

        if len(dictionary) > 0:
            lda_model = gensim.models.LdaMulticore(docs_bow.values(), id2word=dictionary, workers=4, passes=100)
            logger.info("Salvando modelo gerado...")
            lda_model.save(config['paths']['lda'])
