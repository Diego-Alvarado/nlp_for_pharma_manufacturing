import logging
import numpy as np

from time import perf_counter
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary, MmCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_lda_vectors(corpus, lda_model):
    """Calculate LDA vector for a given corpus using gensim lda model.

    Args:
        corpus (Mmcorpus): corpus saved as a sparse coordinate Matrix Market format
        lda_model (gendim model): Lda model

    Yields:
        numpy array [generator]: matrix of LDA topic probabilities for a set of documents.
    """
    test = lda_model[corpus]
    for d in test:
        yield np.array([x[1] for x in d])

if __name__ == "__main__":
    
    dictionary = Dictionary.load(r"training_files\vocab-f-62k.gensim")
    corpus = MmCorpus(r"training_files\corpus-p.mm")
    corpus_len = len(corpus)
    lda = LdaMulticore.load(r"lda2-60t.gensim")

    start = perf_counter()
    X = np.zeros(shape=(corpus_len, 60)) 
    futures = get_lda_vectors(corpus=corpus, lda_model=lda)
    for i, v in enumerate(futures):
        X[i,:] = v
        end = perf_counter() - start
        print(f"Elapsed Time Doc #{i}: {end/60:.2f} minutes", end="\r")
    
    print(f"Elapsed time: {end:.2f} seconds")

    np.save("lda_vectors", X)
    Y = np.load("lda_vectors.npy")
    print(np.allclose(X, Y))
