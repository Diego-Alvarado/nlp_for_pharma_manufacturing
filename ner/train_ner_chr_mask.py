import json
import warnings
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

import keras
from keras.layers import LSTM, Input, Bidirectional, Dropout, \
    BatchNormalization, Conv1D, Activation, TimeDistributed, \
        Embedding, Concatenate, SpatialDropout1D
from keras.models import Model
from keras.utils import pad_sequences, set_random_seed, plot_model
from keras.callbacks import CSVLogger, Callback
from gensim.models.fasttext import load_facebook_model, FastText
from sklearn.model_selection import train_test_split
from seqeval import metrics # package to evaluate ner model performance

def text_vectorizer(sentences: list[list[str]], embeddings: FastText, chr_to_id: dict, maxlen: int =113, max_len_chr: int =22) -> tuple[np.ndarray]:
    # word vectorization:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pad_vect = embeddings.wv.get_vector("", norm=True) 
    word_vectors = [sent[:maxlen] if len(sent) >= maxlen else sent + [""] * (maxlen - len(sent)) for sent in sentences]
    
    # mask vectors
    mask_vectors = [[True if x != "" else False for x in sent] for sent in word_vectors]
    mask_vectors = np.array(mask_vectors, dtype=bool)
    
    # word vectors
    word_vectors = [[embeddings.wv.get_vector(w, norm=True) if w != "" else pad_vect for w in sent] for sent in word_vectors]
    word_vectors = np.array(word_vectors)
    
    # character vectorization:
    chr_vectors = [[[chr_to_id[c] if c in chr_to_id else 1 for c in w] for w in sent] for sent in sentences]
    chr_vectors = [sent[:maxlen] if len(sent) > maxlen else sent + [[0] * max_len_chr] * (maxlen - len(sent)) for sent in chr_vectors]
    chr_vectors = [pad_sequences(seq, maxlen=max_len_chr, truncating="post", padding="post", value=0) for seq in chr_vectors]
    chr_vectors = np.array(chr_vectors)
    
    return word_vectors, chr_vectors, mask_vectors

def outputs(tag_to_id: dict, tags: list[list[str]], pad_sentence: int = 104):
    """
    Convert list of tags to id.
        pad_sentence: sentence max_length
        tag_to_id: dictionary with tag_id
        tags: list of token's labels
    
    Return array of tags_id with shape (Number of sentences, pad_sentence). 
    Sentences with lenght higher than padding are truncated. 
    """
    tag_to_idx = [[tag_to_id[t] for t in tag] for tag in tags]
    tag_to_idx = pad_sequences(tag_to_idx, maxlen=pad_sentence, value=0, padding='post', truncating='post')

    return tag_to_idx

# build model

# Model archiquecture
def create_model(max_len: int, max_chr_len: int, nchar: int,
                 num_labels: int, emb_dim: int,  
                 dropout: float, lstm_units: int, png_name: str) -> Model:
    """
        max_len: sentence max_length
        num_labels: max_length
        emb_dim: dimension of pretrained embeddins
        filter and kernel_size: conv layer hyperparameters
        dropout_rate
        lstm_units: number of unit in BiLSTM layer
        png_file: file name for model architecture
    """
    set_random_seed(202)
    
    # Inputs are calculated using gensim library and then used as inputs in token classification
    word_en = Input(shape=(max_len, emb_dim,), dtype=tf.float32, name="fasttext_embeddings")
    mask = Input(shape=(max_len,), dtype=tf.bool, name="mask")
    
    
    out = Conv1D(filters=512, kernel_size=1, padding='valid')(word_en)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)

    # input and embeddings for characters
    char_in = Input(shape=(max_len, max_chr_len,), name="chr_in")
    emb_char = TimeDistributed(Embedding(input_dim=nchar + 2, output_dim=25,
                            input_length=max_chr_len, mask_zero=True))(char_in)
    char_enc = TimeDistributed(Bidirectional(LSTM(units=25, return_sequences=False)))(emb_char)
    
    
    out = Concatenate()([out, char_enc])
    out = Dropout(dropout)(out)
    out = Bidirectional(LSTM(lstm_units, return_sequences=True, activation='tanh'))(out, mask=mask)
    out = SpatialDropout1D(0.3)(out)
    
    base = Model(inputs=[word_en, char_in, mask], outputs=out)

    model = tfa.text.CRFModelWrapper(base, units=num_labels)
    base.summary()
    
    # Save base model architecture as png
    plot_model(base, png_name, show_shapes=True)
    
    return model

class EarlyStoppingAtMaxF1Score(Callback):
    """Stop training when the f1-score is at its max, i.e. f1-score stops increasing.

    Arguments:
        patience: Number of epochs to wait after max has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, model, X, y, id_to_tag, beta=1.0, patience=0):
        super(EarlyStoppingAtMaxF1Score, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.model = model
        self.X = X
        self.y = y
        self.id_to_tag = id_to_tag
        self.beta = beta

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        y_true = self.y
        
        y_pred = [[id_to_tag[i] for i in (t)] for t in y_pred]
        results = []
        for a, b in zip(y_pred, y_true):
            if len(a) >= len(b):
                results.append(a[:len(b)])
            else:
                diff = len(b) - len(a) 
                c = a + ['O'] * diff
                results.append(c)

        precision = metrics.precision_score(y_true, results, average='micro')
        recall = metrics.recall_score(y_true, results, average='micro')

        f1_score = ((1+self.beta**2)*precision*recall)/(self.beta*precision+recall+1e-8)
        
        print(f'val_precision: {precision:.4f} - val_recall: {recall:.4f} - val_f1_score: {f1_score:.4f}')
        logs['f1_score'] = f1_score
        logs['precision'] = precision
        logs['recall'] = recall

        current = logs.get("f1_score")
        if np.greater(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

# training model
def model_training(model: Model, training_data: list, dev_data: list, lr: float, wd: float, id_to_tag: dict,
                   batch_size: int = 16, epochs: int = 200, patience: int = 20) -> pd.DataFrame:
    """
    Train model.
        model: tensorflow model to train
        training_data: list [X_train, y_train]
        dev_data: development set list [X_dev, y_dev]
        lr: learning rate 
        wd: weight decay in AdamW optimizer

    Return tf model, log csv file, and classification report for dev_data using the best model
    """
    X_t = training_data[0]
    X_d = dev_data[0]

    y_t = training_data[1]
    y_d = dev_data[1]

    callback = [EarlyStoppingAtMaxF1Score(model = model, X=X_d, y=dev['tag'], id_to_tag=id_to_tag, patience=patience),
                CSVLogger(log_file, append=True)]

    opt = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=lr, clipnorm = 5.)
    model.compile(optimizer=opt)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        history = model.fit(x=X_t, y=y_t, validation_data=(X_d, y_d), epochs=epochs, batch_size=batch_size, callbacks=callback)

    # classification report dev_set for best model    

        pred = model.predict(X_d)
        y_pred = [[id_to_tag[i] for i in (t)] for t in pred]
        results = []
        for a, b in zip(y_pred, dev['tag']):
            if len(a) >= len(b):
                results.append(a[:len(b)])
            else:
                diff = len(b) - len(a) 
                c = a + ['O'] * diff
                results.append(c)

        r = pd.DataFrame(metrics.classification_report(dev['tag'], results, output_dict=True)).T
        r.to_csv(report_file)

    return r

# Load embeddings
if __name__ ==  "__main__":
    
    ft = load_facebook_model('fast-pharma-208k300-5.bin')
    
    # load ner dataset
    df = pd.read_csv(r'training_set.tsv', sep="\t", encoding="utf-8")
    df['word'] = df['word'].astype(str)
    df['tag'] = df['tag'].str.upper()

    # create dictionaries
    tag_to_id = {v: k for k, v in enumerate(df['tag'].unique())}
    id_to_tag = {k: v for v, k in tag_to_id.items()}

    with open("id2tag.json", "w") as f:
        json.dump(id_to_tag, f)

    chr_map = set([c for w in df['word'] for c in w if w])
    chr_to_id = {v: k + 2 for k, v in enumerate(chr_map)}
    chr_to_id["[PAD]"] = 0
    chr_to_id["[UNK]"] = 1
    id_to_chr = {k: v for v, k in chr_to_id.items()}
    n_chars = len(chr_map)

    with open("chr_to_id.txt", encoding="utf-8", mode="w") as f:
        for k, v in id_to_chr.items():
            f.write(f"{k} {v}\n")

    # Group tokens by sentence_id and discard sentences with less than 2 tokens

    sentences = df.groupby(['sentence_id']).agg({'word': list, 'tag': list, "label": np.mean})
    sentences.reset_index(drop=True, inplace=True)
    sentences["sent_len"] = [len(p.split()) for p in sentences["word"]]
    sentences = sentences.loc[sentences["sent_len"] >= 2]
    
    # Split data in training, test and development set. 
    # At the first split, sentences are taken maintaing the distribution of sentence cluster (label).

    training, test = train_test_split(sentences , test_size=0.2, random_state=2121, stratify=sentences["label"].values)
    test, dev = train_test_split(test , test_size=0.5, random_state=2121)

    # Convent words to vector with sentences max_len = 113

    max_len = 113
    max_len_char = 22
    emb_dim = 300
    
    X_train = text_vectorizer(training['word'], embeddings=ft, chr_to_id=chr_to_id, maxlen=max_len)
    X_dev = text_vectorizer(dev['word'], embeddings=ft, chr_to_id=chr_to_id, maxlen=max_len)
    X_test = text_vectorizer(test['word'], embeddings=ft, chr_to_id=chr_to_id, maxlen=max_len)

    y_train = outputs(tag_to_id, training['tag'], max_len)
    y_dev = outputs(tag_to_id, dev['tag'], max_len)
    y_test = outputs(tag_to_id, test['tag'], max_len)
    
    # model hyperparameters
    
    num_labels = len(tag_to_id)
    wd = 10**-4.130538 
    lr = 10**-2.856609 
    dropout = 0.7
    
    file_model = f"BiLSTM-chr-dropout-{dropout}"
    
    # model training    
    log_file = "log_" + str(file_model) + ".csv"
    report_file = str(file_model) + ".csv"
    png_file = file_model + ".png"
    
    model = create_model(max_len=max_len, num_labels=len(tag_to_id), max_chr_len=max_len_char,
                        nchar=len(chr_map),
                        emb_dim=emb_dim, dropout=dropout, lstm_units=300,
                        png_name=png_file)
    
    report = model_training(model, [X_train, y_train], [X_dev, y_dev], lr, wd,
                            id_to_tag=id_to_tag, batch_size=16, patience=10)
    
    model.save(f'saved_model/NER-{file_model}')

    # Load model
    model = keras.models.load_model(f'saved_model/NER-{file_model}')

    # Predict values for test_set and assess model performance

    pred = model.predict(X_test, batch_size=128)
    y_pred = [[id_to_tag[i] for i in (t)] for t in pred]
    results = []
    for a, b in zip(y_pred, test['tag']):
        if len(a) >= len(b):
            results.append(a[:len(b)])
        else:
            diff = len(b) - len(a) 
            c = a + ['O'] * diff
            results.append(c)

    report = pd.DataFrame(metrics.classification_report(test['tag'], results, output_dict=True)).T
    print(report)
    report.to_csv("test-" + report_file)



