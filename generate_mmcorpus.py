import csv
import logging
import numpy as np

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus

csv.field_size_limit(922337203)

class MyCorpus:
    def __init__(self, fname, dictionary, Qmax = 424) -> None:
        """Load and transform corpus into BoW representation as a generator

        Args:
            fname (str): file path tsv file - corpus
            dictionary (str): file path gensim dictionary for BoW
            Qmax (int, optional): Maximum number of token to truncate text. To avoid truncation, change value by np.Inf. Defaults to 424.
        """
        self.fname = fname
        self.dictionary = Dictionary.load(dictionary)
        self.Qmax = Qmax
        
    def __iter__(self):
        with open(self.fname, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if len(row["prep_desc"].split()) >= self.Qmax:
                    yield self.dictionary.doc2bow(row["prep_desc"].split()[:self.Qmax])
                else:
                    yield self.dictionary.doc2bow(row["prep_desc"].split())

if __name__ == "__main__":
    
    # Activate logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dict_path = r"training_files\vocab-f-62k.gensim"
    fpath = r"datasets\sect.tsv"
    
    # Corpus serialized using the sparse coordinate Matrix Market format to facilitate training for large files
    df = MyCorpus(fname=fpath, dictionary=dict_path)
    
    corpus = MmCorpus.serialize(fname=r"training_files\corpus-p.mm", corpus=df, id2word=df.dictionary)