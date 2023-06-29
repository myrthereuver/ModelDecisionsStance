import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import logging
import string
import nltk
nltk.download('punkt')
import numpy as np
import time
import os
import random
from argparse import ArgumentParser
from typing import Sequence
import datetime
from sib import SIB
from keras.callbacks import CSVLogger
import re
from sklearn.feature_extraction import _stop_words


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# device = "cuda:0"
#if torch.cuda.is_available() else "cpu"

## BASED ON CODE FROM:
# * https://github.com/IBM/intermediate-training-using-clustering/blob/main/run_experiment.py
# *

def import_df(path, n=None):
    data = pd.read_csv(path)
    df = data.dropna()
    df = df.sample(frac=1, random_state=42)
    if n is not None:
        df = df.sample(n=n,random_state=42)
    return df

def make_lists(df):
    text1 = df["text1"].tolist()
    text2 = df["text2"].tolist()
    labels = df["label"].tolist()
    labels = list(map(int, labels))
    return text1, text2, labels

def encode_data(text1, labels, model_name, model_path, text2=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if text2 is not None:
        encodings = tokenizer(text1,
                              text2,
                              padding="max_length",
                              truncation=True,
                              return_tensors="tf")
        encodings = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    else:
        encodings = tokenizer(text1,
                              padding="max_length",
                              truncation=True,
                              return_tensors="tf")
        encodings = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))

    tokenizer.save_pretrained(model_path)
    return encodings

def remove_stop_words_and_punctuation(texts):
    stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
    escaped_stop_words = [re.escape(stop_word) for stop_word in stop_words]
    regex_pattern = r"\b(" + "|".join(escaped_stop_words) + r")\b"
    # remove stop words
    texts = [re.sub(r" +", r" ", re.sub(regex_pattern, "", str(text).lower())).strip() for text in texts]
    # remove punctuation
    texts = [t.translate(t.maketrans(string.punctuation, ' ' * len(string.punctuation))) for t in texts]
    #texts = [re.sub(r"(\.:'\?!\[()\])", r" ", re.sub(regex_pattern, "", str(text).lower())).strip() for text in texts]
    return [' '.join(t.split()) for t in texts]


def stem(texts):
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=False)
    return [" ".join(stemmer.stem(word).lower().strip()
            for word in nltk.word_tokenize(text)) for text in texts]


def get_embeddings(texts):
    # apply text processing to prepare the texts
    texts = remove_stop_words_and_punctuation(texts)
    texts = stem(texts)

    # create the vectorizer and transform data to vectors
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=10000, stop_words=None, use_idf=False, norm=None)
    vectors = vectorizer.fit_transform(texts)
    return vectors

def dict_cluster(model, texts):
    labels = model.predict(texts)
    clusters = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(texts[n])
        else:
            clusters[item] = [texts[n]]
        n += 1
    return clusters

def print_clusterdict(clusters):
    for item in clusters:
        print("Cluster ", item)
        for i in clusters[item]:
            print(i)
    # return prints


def get_cluster_labels(texts, n_clusters): #From code clustering paper
    embedding_vectors = get_embeddings(texts)
    logging.info('Finished generating embedding vectors')
    algorithm = SIB(n_clusters=n_clusters, n_init=10, n_jobs=-1, max_iter=15, random_state=args.random_seed, tol=0.02)
    clustering_model = algorithm.fit(embedding_vectors)
    logging.info(f'Finished clustering embeddings for {len(texts)} texts into {n_clusters} clusters')
    cluster_labels = clustering_model.labels_.tolist()
    return cluster_labels

def train_model(train_data, batch_size, learning_rate, epochs, num_labels, model_path, save_path, callback_path, log_file, tensorflows, intertrain=False, nli_data=None):
    #cluster_labels = get_cluster_labels(texts, args.clusters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)

    model = TFAutoModelForSequenceClassification.from_pretrained(model_path,
                                                                 num_labels=num_labels,
                                                                 # from_tf=False,
                                                                 from_pt=True,
                                                                 ignore_mismatched_sizes=True)
    model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
                    callback_path,
                    monitor='loss',
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=True,
                    mode='auto',
                    save_freq='epoch',
                    options=None,
                    initial_value_threshold=None,
                    )
    csv_logger = CSVLogger(log_file)

    model.fit(x=train_data.shuffle(1000).batch(batch_size),
              validation_data=None,
              epochs=epochs,
              callbacks=[checkpoints, csv_logger])

    if intertrain==True:
        model.classifier._name = "dummy"  # change classifier layer name so it will not be reused

    configuration = model.config
    model.save_pretrained(save_path)
    model.config.__class__.from_pretrained(model_name).save_pretrained(save_path)
    return model, save_path, configuration


def predict(texts_1, texts_2, model_name, num_labels, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encodings = tokenizer(texts_1,
                          texts_2,
                          padding="max_length",
                          truncation=True,
                          return_tensors="tf")

    logging.info(f'running inference using model {model_name} on {len(texts_1)} text pairs')

    model = TFAutoModelForSequenceClassification.from_pretrained(model_path,
                                                                 num_labels=num_labels)
    predictions = model(encodings)
    logits = predictions.logits.numpy().tolist()
    preds = np.argmax(logits, axis=1)
    return preds

def evaluate(true, pred, model_name, TRAIN, DEV, timing):
    results = []
    labels = [int(x) for x in true]

    model = f"model is {model_name}"
    train_data = f"train data is {TRAIN}"
    test_data = f"test data is {DEV}"

    acc = accuracy_score(true, pred)
    accuracy = f"accuracy is {acc}"

    f1 = precision_recall_fscore_support(labels, pred, average='macro')
    prec = f"precision is {f1[0]}"
    rec = f"recall is {f1[1]}"
    f1 = f"F1 is {f1[2]}"

    report = classification_report(labels, pred)

    time = (f"this took: {timing}")

    results.extend([train_data, test_data, model, accuracy, prec, rec, f1, report, time])
    results_string = "\n".join(results)
    print(results_string)
    return results_string

def write_results(results, filename):
    with open(filename, "a+") as file:
        file.write(results)

################

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--model', type=str, default='albert-base-v2', required=True)
    parser.add_argument('--tensorflows', type=bool, default=True, help="enable loading from Tensorflow models")
    parser.add_argument('--random_seed', type=int, default=42, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--inter_training_task', type=str, default=None)
    parser.add_argument('--inter_training_epochs', type=int, default=1)
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--nli_data', type=str)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--learning_rate', type=int, default=0.00003)

    args = parser.parse_args()
    logging.info(args)

    # set random seed
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    start = time.time()

    TRAIN = args.train_file
    DEV = args.eval_file

    model_name = args.model
    SAVE_path = args.save_path

    RATE=args.learning_rate
    EPOCHS=args.epochs
    BATCH = args.batch
    tensorflows = args.tensorflows

    NUM_LABELS=2

    ###################

    train_df = import_df(TRAIN)
    dev_df = import_df(DEV)
    #test_df = import_df(TEST)

    df_arc_train_data = train_df.where(train_df["dataset"] == "arc_train").dropna()
    df_ibmcs_train_data = train_df.where(train_df["dataset"] == "ibmcs_train").dropna()
    df_perspectrum_train_data = train_df.where(train_df["dataset"] == "perspectrum_train").dropna()
    df_fnc1_train_data = train_df.where(train_df["dataset"] == "fnc1_train").dropna()
    df_iac1_train_data = train_df.where(train_df["dataset"] == "iac1_train").dropna()
    df_semeval2016t6_train_data = train_df.where(train_df["dataset"] == "semeval2016t6_train").dropna()
    df_semeval2019t7_train_data = train_df.where(train_df["dataset"] == "semeval2019t7_train").dropna()
    df_snopes_train_data = train_df.where(train_df["dataset"] == "snopes_train").dropna()
    df_argmin_train_data = train_df.where(train_df["dataset"] == "argmin_train").dropna()
    df_scd_train_data = train_df.where(train_df["dataset"] == "scd_train").dropna()

    df_arc_dev_data = dev_df.where(dev_df["dataset"] == "arc_dev").dropna()
    df_ibmcs_dev_data = dev_df.where(dev_df["dataset"] == "ibmcs_dev").dropna()
    df_perspectrum_dev_data = dev_df.where(dev_df["dataset"] == "perspectrum_dev").dropna()
    df_fnc1_dev_data = dev_df.where(dev_df["dataset"] == "fnc1_dev").dropna()
    df_iac1_dev_data = dev_df.where(dev_df["dataset"] == "iac1_dev").dropna()
    df_semeval2016t6_dev_data = dev_df.where(dev_df["dataset"] == "semeval2016t6_dev").dropna()
    df_semeval2019t7_dev_data = dev_df.where(dev_df["dataset"] == "semeval2019t7_dev").dropna()
    df_snopes_dev_data = dev_df.where(dev_df["dataset"] == "snopes_dev").dropna()
    df_argmin_dev_data = dev_df.where(dev_df["dataset"] == "argmin_dev").dropna()
    df_scd_dev_data = dev_df.where(dev_df["dataset"] == "scd_dev").dropna()

    # df_arc_test_data = test_df.where(test_df["dataset"] == "arc_test").dropna()
    # df_ibmcs_test_data = test_df.where(test_df["dataset"] == "ibmcs_test").dropna()
    # df_perspectrum_test_data = test_df.where(test_df["dataset"] == "perspectrum_test").dropna()
    # df_fnc1_test_data = test_df.where(test_df["dataset"] == "fnc1_test").dropna()
    # df_iac1_test_data = test_df.where(test_df["dataset"] == "iac1_test").dropna()
    # df_semeval2016t6_test_data = test_df.where(test_df["dataset"] == "semeval2016t6_test").dropna()
    # df_semeval2019t7_test_data = test_df.where(test_df["dataset"] == "semeval2019t7_test").dropna()
    # df_snopes_test_data = test_df.where(test_df["dataset"] == "snopes_test").dropna()
    # df_argmin_test_data = test_df.where(test_df["dataset"] == "argmin_test").dropna()
    # df_scd_test_data = test_df.where(test_df["dataset"] == "scd_test").dropna()

    list_datasets_train_all = [df_arc_train_data, df_perspectrum_train_data, df_fnc1_train_data,
                               df_iac1_train_data, df_semeval2016t6_train_data,
                               df_semeval2019t7_train_data,
                               df_scd_train_data, df_ibmcs_train_data, df_iac1_train_data,
                               df_semeval2016t6_train_data, df_semeval2019t7_train_data,
                               df_snopes_train_data, df_argmin_train_data]

    list_datasets_dev_all = [df_arc_dev_data, df_perspectrum_dev_data, df_fnc1_dev_data,
                             df_iac1_dev_data, df_semeval2016t6_dev_data, df_semeval2019t7_dev_data,
                             df_scd_dev_data, df_ibmcs_dev_data, df_iac1_dev_data, df_semeval2016t6_dev_data,
                             df_semeval2019t7_dev_data,
                             df_snopes_dev_data, df_argmin_dev_data]

    # list_datasets_test_all = [df_arc_test_data, df_perspectrum_test_data, df_fnc1_test_data,
    #                           df_iac1_test_data, df_semeval2016t6_test_data,
    #                           df_semeval2019t7_test_data,
    #                           df_scd_test_data, df_ibmcs_test_data, df_iac1_test_data,
    #                           df_semeval2016t6_test_data, df_semeval2019t7_test_data,
    #                           df_snopes_test_data, df_argmin_test_data]

    list_datasets_str_all = ['df_arc_train_datadata', 'df_ibmcs_train_data', 'df_perspectrum_train_data', 'df_fnc1_train_data', 'df_iac1_train_data', 'df_semeval2016t6_train_data', 'df_semeval2019t7_train_data', 'df_snopes_train_data', 'df_argmin_train_data', 'df_scd_train_data']

    ########
    model_dir = f"model_{model_name}"
    data_dir = f"{TRAIN}"

    now = datetime.datetime.now()
    date_month = now.strftime("%d-%B")

    output_dir = os.path.join(model_name, data_dir, "output/")
    model_save = f"{date_month}_{model_name}_TRAIN-size:100"

    # ######################
    # print("CLUSTER PRINT")
    # all_texts = train1_sub + train2_sub
    # embedding_vectors = get_embeddings(all_texts)
    # algorithm = SIB(n_clusters=args.clusters, n_init=10, n_jobs=-1, max_iter=15, random_state=args.random_seed, tol=0.02)
    # clustering_model = algorithm.fit(embedding_vectors)
    # clusters = dict_cluster(clustering_model, embedding_vectors)
    # print_clusterdict(clusters)
    # #######################
    n = 0
    for data in zip(list_datasets_train_all, list_datasets_dev_all):
        data_name = list_datasets_str_all[n]
        n = n + 1

        train_text1, train_text2, train_labels = make_lists(data[0].sample(n=100, random_state=42))
        dev_text1, dev_text2, dev_labels = make_lists(data[1])


        train_encodings = encode_data(text1=train_text1,
                                  text2=train_text2,
                                  labels=train_labels,
                                  model_name=model_name,
                                  model_path=SAVE_path)


        if args.inter_training_task == "cluster":
            all_texts = train1_sub + train2_sub
            print("INPUT")
        #print(all_texts[:2])
            cluster_labels = get_cluster_labels(all_texts, args.clusters)
            logging.info(f'Clustering {len(all_texts)} unlabeled texts into {args.clusters} clusters')

            cluster_save = "cluster_model_fulldata/"

            cluster_encodings = encode_data(text1=all_texts,
                                        labels=cluster_labels,
                                        model_name=model_name,
                                        model_path=cluster_save)

            print("CLUSTERING")

            cluster_model, cluster_model_path, cluster_model_config = train_model(cluster_encodings,
                                                                                batch_size=BATCH,
                                                                                learning_rate=RATE,
                                                                                epochs=args.inter_training_epochs,
                                                                                num_labels=len(cluster_labels),
                                                                                model_path=model_name,
                                                                                save_path=cluster_save,
                                                                                log_file="cluster_weights.hdf5",
                                                                                callback_path=f"cluster_logs_per_epoch_{model_name}_train-{TRAIN}_dev-{DEV}.txt",
                                                                                intertrain=True)


            model, model_path, model_config = train_model(train_encodings,
                                                     batch_size=BATCH,
                                                     learning_rate=RATE,
                                                     epochs=EPOCHS,
                                                     num_labels=NUM_LABELS,
                                                     model_path=cluster_model_path,
                                                     save_path=SAVE_path,
                                                     callback_path="fullseeds_clusterStance_weights.hdf5",
                                                     log_file=f"fullseeds_clusterStance_logs_per_epoch_{model_name}_train-{TRAIN}_dev-{DEV}.txt",
                                                     intertrain=False)

        elif args.inter_training_task == "nli":
            print("NLI")
            df_nli = import_df(args.nli)

            nli_save = "nli_model/"

            nli_train1, nli_train2, nli_train_labels_sub = make_lists(df_nli)

            nli_encodings = encode_data(nli_text1,
                                    train_text2,
                                    nli_labels,
                                    model_name,
                                    nli_save)

            nli_model, nli_model_path, nli_model_config = train_model(nli_encodings,
                                                                  batch_size=BATCH,
                                                                  learning_rate=RATE,
                                                                  epochs=args.inter_training_epochs,
                                                                  num_labels=NUM_LABELS,
                                                                  model_path=model_name,
                                                                  save_path=nli_save,
                                                                  callback_path="nli_weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                                                  log_file=f"nli_logs_per_epoch_{model_name}_train-{TRAIN}_dev-{DEV}.txt",
                                                                  intertrain=True)

            model, model_path, model_config = train_model(train_encodings,
                                                      batch_size=BATCH,
                                                      learning_rate=RATE,
                                                      epochs=EPOCHS,
                                                      num_labels=NUM_LABELS,
                                                      model_path=nli_model_path,
                                                      save_path=SAVE_path,
                                                      callback_path="nliStance_weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                                      log_file=f"nliStance_logs_per_epoch_{model_name}_train-{TRAIN}_dev-{DEV}.txt")
        else:
            print("TRAIN WITHOUT INTERTRAIN")
            model, model_path, model_config = train_model(train_encodings,
                                                      batch_size=BATCH,
                                                      learning_rate=RATE,
                                                      epochs=EPOCHS,
                                                      num_labels=NUM_LABELS,
                                                      model_path=model_name,
                                                      save_path=SAVE_path,
                                                      tensorflows=tensorflows,
                                                      callback_path="weights.hdf5",
                                                      log_file=f"logs_per_epoch_{model_name}_train-{TRAIN}_dev-{DEV}.txt",
                                                      intertrain=False)


        predictions = predict(dev_text1,
                    dev_text2,
                    model_name=model_path,
                    num_labels=NUM_LABELS)

        stop = time.time()
        timing = stop-start
        time_processed = datetime.timedelta(seconds=timing)
        print(f"this took: {time_processed}")

        results = evaluate(predictions, dev_labels, model_name, TRAIN, DEV, time_processed)

        write_results(results,
                  f"100ex_model-{model_name}_traindata-{len(train1_sub)}.txt")

