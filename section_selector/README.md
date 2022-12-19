# Section selector for pharmaceutical manufacturing patents

## Description

Code and files to train a hybrid approach based on LDA+kMeans techniques to select the sections of the patents that contain information on either primary or secondary pharmaceutical manufacturing, with particular emphasis on small molecules.

## Files

The files employed to train the models are contained in training_files folder. The corpus is compressed in 7z format. On decompressing, the preprocessed corpus in Matrix Market format will be found and ready to train LDA (train_lda.py). With the best model (num_topics = 60), LDA vectors are calculated (pred_lda_vect.py), which are used to run minibatch kMeans (train_kmeans.py).

For LDA training, perplexities and keywords for the best model can be seen in results_best_lda. Clustering performance and label assignation per each cluster are stored in results_cluster folder.

