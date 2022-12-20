import csv
import logging
import numpy as np

from time import perf_counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize
            
def _model_performance(X, labels, subsample: int = None):
        if subsample:
            s = np.random.randint(len(labels), size=subsample)
            X_sample = X[s]
            y_sample = labels[s]
            db = davies_bouldin_score(X_sample, y_sample)
            sl = silhouette_score(X_sample, y_sample)
            return db, sl
        
        else:
            db = davies_bouldin_score(X, labels)
            sl = silhouette_score(X, labels)
            return db, sl

if __name__ == "__main__":
    # Activate logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # Load LDA vectors
    X_lda = np.load("lda_vectors.npy")
    
    # Train minibatch kMeans varying data transformation and number of clusters
    for dist in ["cosine", "hellinger"]:
        
        if dist == "cosine":
            X = normalize(X_lda)
        else:
            X = (X_lda / X_lda.sum(1).reshape(-1, 1))**0.5

        for k in range(5, 61, 5):
            print("Training model...")
            start = perf_counter()
            kmean = MiniBatchKMeans(n_clusters=k, random_state=100, batch_size=4096)
            labels = kmean.fit_predict(X)
            end = perf_counter() - start
            print(f"Elapsed time: {end/60:.2f}")
            
            print("Training model...")        
            with open("performance.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["distance", "k", "db", "slt", "time"])
                if k == 5:
                    writer.writeheader()
                for _ in range(7):
                    db, sl = _model_performance(X, labels, 10000)    
                    writer.writerow({"distance": dist,
                                    "k": k, 
                                    "db": db, 
                                    "slt": sl,
                                    "time": end})
            end = perf_counter() - start
            print(f"Elapsed time model {k}: {end/60:.2f}")
                    
