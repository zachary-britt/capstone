import sys
#sys.path.insert(0, '/home/zachary/dsi/sense2vec')
#sys.path.insert(0, "/home/zachary/anaconda3/lib/python3.6/site-packages/spacy")
from gensim.models import Word2Vec
import plac
import os

@plac.annotations(
    rebuild=("Rebuild vectorizer", "flag", "r", bool)
)
def main(rebuild=False):
    if rebuild:
        os.system('home/zachary/dsi/capstone/src/sense2vec_script')
    else:
        os.system('echo using saved vectorizer')

    model_path = 'home/zachary/dsi/capstone/data/articles2/trained_model'
    gensim_model = Word2Vec.load(model_path)


if __name__ == '__main__':
    plac.call(main)
