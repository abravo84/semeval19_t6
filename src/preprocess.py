import re, csv, random, html
from utils import save_obj
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50

def prepocess_organizers_dataset(add_path, add_proc_path):
    count = 0
    total = 0

    label_tranf = {"CAG": "OFF",
                   "OAG": "OFF",
                   "NAG": "NOT"}
    out = open(add_proc_path, "w")
    # out.write("id\ttweet")
    with open(add_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweet = row['tweet'].replace("\n", " ")

            total += 1
            if len(tweet.split(" ")) > 55:
                continue
            if tweet.startswith("RT"):
                continue
            if " RT " in tweet:
                continue
            if not len(tweet):
                continue
            count += 1

            tweet = re.sub(r"http\S+", "URL", tweet)
            tweet = re.sub('@[^\s]+', '@USER', tweet)

            print(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']])

            out.write(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']] + "\n")

    print(total)
    print(count)
    out.close()

def prepocess_additional_dataset(add_path, add_proc_path):
    count = 0
    total = 0

    label_tranf = {"0": "OFF",
                   "1": "OFF",
                   "2": "NOT"}
    out = open(add_proc_path, "w")
    # out.write("id\ttweet")
    with open(add_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweet = row['tweet'].replace("\n", " ")
            total += 1
            if tweet.startswith("RT"):
                continue
            if " RT " in tweet:
                continue
            if not len(tweet):
                continue
            count += 1
            #print(row['id'], row['class'], tweet)

            tweet = re.sub(r"http\S+", "URL", tweet)
            tweet = re.sub('@[^\s]+', '@USER', tweet)

            print(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']])

            out.write(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']] + "\n")

    print(total)
    print(count)
    out.close()


def preprocess_task3(train_file, new_path):
    # Create data task3
    """
    out = open(new_path, "w")
    out.write("id" + "\t" + "tweet" + "\t" + "subtask_a" + "\t" + "subtask_b" + "\t" + "subtask_c" + "\n")
    original_train = []
    duplicated_train = []
    fd = open(train_file, "r")
    read = csv.DictReader(fd, dialect="excel-tab")
    for row in read:
        if row["subtask_b"] == "TIN":

            nl = row["id"] + "\t" + row["tweet"] + "\t" + row["subtask_a"] + "\t" + row["subtask_b"] + "\t" + row[
                "subtask_c"]
            out.write(nl+"\n")

    fd.close()
    out.close()
    """

    out = open(new_path, "w")
    original_train = []
    duplicated_train = []
    fd = open(train_file, "r")
    read = csv.DictReader(fd, dialect="excel-tab")
    for row in read:
        if row["subtask_a"] == "NOT":
            continue

        nl = row["id"] + "\t" + row["tweet"] + "\t" + row["subtask_a"] + "\t" + row["subtask_b"] + "\t" + row[
            "subtask_c"]

        original_train.append(nl)

        if row["subtask_b"] == "UNT":
            duplicated_train.append(nl)

    random.shuffle(original_train)

    test_count = int(len(original_train) * 0.15)
    print("test_count: " + str(test_count))
    print("duplicated_train: " + str(len(duplicated_train)))

    test_data = original_train[:test_count]
    original_train = original_train[test_count:]

    perfect_duplicated_data = []

    print("test_data: " + str(len(test_data)))
    print("original_train: " + str(len(original_train)))

    for l in duplicated_train:
        if not l in test_data:
            perfect_duplicated_data.append(l)

    print("perfect_duplicated_data: " + str(len(perfect_duplicated_data)))

    original_train = original_train + perfect_duplicated_data + perfect_duplicated_data + perfect_duplicated_data

    random.shuffle(original_train)
    random.shuffle(original_train)

    original_train = test_data + original_train

    out.write("id" + "\t" + "tweet" + "\t" + "subtask_a" + "\t" + "subtask_b" + "\t" + "subtask_c" + "\n")

    for nl in original_train:
        out.write(nl + "\n")

    out.close()
    fd.close()



def get_pretrained_embedding(emb_path, word_index):
    embeddings_index = {}
    f = open(emb_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.random.uniform(-0.8, 0.8, (len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def clean_tweet(tweet_text):
	tweet_text = tweet_text.replace('—', " ")#.replace("'", "’")
	tweet_text = ' '.join(tweet_text.split())
	return tweet_text.strip()



def text_processing(nlp, text):
    tweet_text = clean_tweet(text)

    tweet_text = tweet_text.replace("'", "&&&").replace("’", "&&&").replace("-", "&#&")

    doc = nlp(tweet_text)
    print(text)

    for token in doc:
        print(str(token).replace("&&&", "'").replace("&#&","-"))





def replace_tweet(tweet_text):

    tweet_text = html.unescape(tweet_text)

    return tweet_text.replace("'", "QUOTE_SYMBOL").replace("‘", "QUOTE_SYMBOL").replace("’", "QUOTE_SYMBOL").replace("-", "HYPH_SYMBOL").replace(";", " ").replace("#", "HASHTAG_SYMBOL")

def unreplace_tweet(tweet_text):

    return tweet_text.replace("QUOTE_SYMBOL", "'").replace("HYPH_SYMBOL", "-").replace("HASHTAG_SYMBOL", "#").replace("EMOJI_SYMBOL","#&").lower()


def get_preprocessed_labels(path, label_type="subtask_a", label_dict= {"OFF": 1}):
    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")

    labels = []

    for row in read:

        label = label_dict.get(row[label_type], 0)
        labels.append(label)

    return labels




def get_word_index(path, nlp):

    fd = open(path, "r")
    read = csv.DictReader(fd, dialect="excel-tab")
    word_set = set()
    i=0
    for row in read:
        i += 1
        tweet_text = clean_tweet(row["tweet"])
        tweet_text = replace_tweet(tweet_text)
        doc = nlp(tweet_text)
        for token in doc:
            word = unreplace_tweet(str(token))
            word_set.add(word)

    word_index = {}
    i = 0
    for tok in word_set:
        i += 1
        word_index[tok] = i

    fd.close()
    return word_index

def get_word_index_from_file(path):
    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")
    word_index = {}
    for row in read:
        word_index[row["word"]] = int(row["index"])
    fd.close()
    return word_index

def get_preprocessed_tweets(path, nlp, word_index, label_type="subtask_a", label_dict= {"OFF": 1}, max_instances=0):

    print(path)

    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")

    texts = []

    labels = []

    i=0

    for row in read:

        sent = []
        i += 1

        tweet_text = clean_tweet(row["tweet"])
        if "test" in label_type:
            label=row["id"]
        else:
            label = label_dict.get(row[label_type], 0)
        labels.append(label)
        tweet_text =replace_tweet(tweet_text)

        doc = nlp(tweet_text)

        for token in doc:
            word = unreplace_tweet(str(token))
            sent.append(word)

        texts.append(sent)
        if i == max_instances:
            break

    sequences = []
    for sent in texts:
        seq = []
        for word in sent:
            i = word_index.get(word, 0)
            if i:
                seq.append(i)
        sequences.append(seq)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    if not "test" in label_type:
        labels = to_categorical(np.asarray(labels))
    fd.close()
    return data, labels

def create_data_task3(new_path, vocab_path, word_index_pkl_path, embedding_matrix_path, emb_path, data_path, labels_path, nlp):
    #WORD_INDEX
    word_index = get_word_index(vocab_path, nlp)
    word_index_path = "../data/word_index_train_test_task3.tsv"
    out = open(word_index_path, "w")
    out.write("word" + "\t" + "index" + "\n")
    for k in sorted(word_index):
        out.write(k + "\t" + str(word_index[k]) + "\n")
    out.close()
    save_obj(word_index, word_index_pkl_path)


    #embedding_matrix
    embedding_matrix = get_pretrained_embedding(emb_path, word_index)
    save_obj(embedding_matrix, embedding_matrix_path)




    label_dict = {"OTH":0,
                       "IND":1,
                       "GRP":2}
    train_file = "../data/offenseval-training-v1_task3.tsv"
    data, labels = get_preprocessed_tweets(train_file, nlp, word_index, "subtask_c", label_dict)
    save_obj(data, data_path)
    save_obj(labels, labels_path)





def create_data_test_task3(data_path, ids_test_path, testing_path, word_index, nlp):
    data_test, ids_test = get_preprocessed_tweets(testing_path, nlp, word_index, "test")
    save_obj(data_test, data_path)
    save_obj(ids_test, ids_test_path)

if __name__ == '__main__':
    add_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/organizers_data.csv"
    add_proc_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/organizers_data_processed.tsv"
    prepocess_organizers_dataset(add_path, add_proc_path)

    add_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/labeled_data.csv"
    add_proc_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/labeled_data_processed.tsv"
    prepocess_additional_dataset(add_path, add_proc_path)