import os
import logging
import argparse

from data import Corpus

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#
parser = argparse.ArgumentParser(description='Tokenizing non-English corpora')
parser.add_argument('--glove_emb', action='store_true',
                        help='use pretrained GloVe embeddings')
args = parser.parse_args()

BASE_DIR = os.path.abspath(os.path.join(__file__, '../..'))

for task in ["ta_wiki", "de_wiki", "he_wiki", "it_wiki", "ru_wiki", "en_wiki"]:
    _logger.info(f"Processing: {task}")
    try:
        pathname = os.path.join(BASE_DIR, f"data/{task}/")
        corpus = Corpus(pathname, args)
    except:
        _logger.info(f"No dataset folder found for {task}!")
