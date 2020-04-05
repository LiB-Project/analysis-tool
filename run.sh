pipenv install

sh generate-stubs.sh

pipenv run \
  python -m nltk.downloader punkt averaged_perceptron_tagger stopwords wordnet

pipenv run python server.py
