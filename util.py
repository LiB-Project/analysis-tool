import logging
import os
import pickle
import gensim
import yaml

logger = logging.getLogger(__name__)
# Load configuration
with open('config.yml') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)
fp.close()


class Util:
    @staticmethod
    def save_docs_preprocessed(list_docs):
        with open(config['paths']['docs_processed'], 'wb') as fp:
            pickle.dump(list_docs, fp)

    @staticmethod
    def save_docs_bow(bows):
        with open(config['paths']['docs_bow'], 'wb') as fp:
            pickle.dump(bows, fp)

    @staticmethod
    def read_docs_bow():
        with open(config['paths']['docs_bow'], 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def read_docs_preprocessed():
        with open(config['paths']['docs_processed'], 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def docs_preprocessed_exists():
        return os.path.exists(config['paths']['docs_processed'])

    @staticmethod
    def load_dictionary():
        if not os.path.exists(config['paths']['dictionary']):
            logger.info("Dicionario nao existe, criando...")
            return gensim.corpora.Dictionary()
        else:
            return gensim.corpora.Dictionary.load(config['paths']['dictionary'])

    @staticmethod
    def load_lda_model():
        return gensim.models.LdaMulticore.load(config['paths']['lda'])
