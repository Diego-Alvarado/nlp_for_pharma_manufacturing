import csv
import logging
import numpy as np

from time import perf_counter
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
   
    # Load dictionary
    dictionary = Dictionary.load(r"training_files\vocab-f-62k.gensim")
    
    # Load corpus - corpus-p (truncation)
    corpus = MmCorpus(r"training_files\corpus-p.mm")
    corpus_size = len(corpus)
    
    # split data in training and test set
    np.random.seed(200)
    test = np.random.randint(0, corpus_size, size= int(0.1 * corpus_size))
    train = np.arange(corpus_size)[np.isin(np.arange(corpus_size), test, invert=True)]
    
    # performance file name
    perf_name = "perplexity.csv"

    for k in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]:
        # shuffle data
        np.random.shuffle(train)
        np.random.shuffle(test)
        
        training_set = corpus[ train]
        test_set = corpus[test]
        
        # Save logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh = logging.FileHandler(f'lda2-{k}t.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Train model
        lda = LdaMulticore(corpus=training_set, id2word=dictionary, alpha="asymmetric",
                        workers=16, num_topics=k, random_state=32, passes=1,
                        chunksize=4096, minimum_probability=0.)

        print("Saving model...")
        lda.save(f"lda2-{k}t.gensim")

        print("Calculating perplexity...")
        lda_perplexity = lda.log_perplexity(test_set)

        # Save perplexity per number of topics
        with open(perf_name, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["log-perplexity", "k"])
            if k == 5:
                writer.writeheader()
            writer.writerow({"log-perplexity": lda_perplexity, "k": k})

        # Save keywords per each LDA model built
        with open(f"keywords-lda2-{k}t.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["topic", "rank", "keyword", "prob"])
            writer.writeheader()
            for topic, keywords in lda.show_topics(num_topics=-1, num_words=20, formatted=False):
                for rank, (key, prob) in enumerate(keywords):
                    writer.writerow({"topic": topic, 
                                    "rank": rank, 
                                    "keyword": key, 
                                    "prob": prob})

        logger.removeHandler(fh)

