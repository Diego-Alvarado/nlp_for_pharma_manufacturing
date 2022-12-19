# Section selector for pharmaceutical manufacturing patents

## Files description

The files employed to train the models are contained in training_files folder. The corpus is compressed in 7z format. On decompressing, the preprocessed corpus in Matrix Market format will be found and ready to train LDA (train_lda.py). With the best model (num_topics = 60), LDA vectors are calculated (pred_lda_vect.py) which are used to run minibatch kMeans (train_kmeans.py).

For LDA training, perplexities and keywords for the best model can be seen in results_best_lda. Clustering performance and labels assignation per each cluster are store in results_cluster folder.

