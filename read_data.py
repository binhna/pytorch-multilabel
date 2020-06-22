import pandas
import re
import logging
import os
import torch
import numpy as np
import nltk.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json
import MeCab

mecab_tokenizer = MeCab.Tagger(
    f"-Ochasen -d /home/aimenext/nlp_data/neologd_dic")

with open('/home/aimenext/nlp_data/stopwords-iso.json', 'r') as f:
    stopwords = json.load(f)
    stopwords = stopwords['ja']
stopwords.extend(list('、。・；’：”＜＞？｛｝＝＋ー）（＊＆＾％＄＠！〜｀●△【】■×)('))

def tokenizer(sentence):
    sentence = mecab_tokenizer.parse(sentence)
    words = [i.split('\t')[0] for i in sentence.split('\n')[:-2]]
    return words

class Dictionary:
    def __init__(self):
        self.word2idx = {'<pad>': 0}
        self.idx2word = ['<pad>']

    def add_vocab(self, word):
        if word not in self.word2idx:
            index = len(self.idx2word)
            self.word2idx[word] = index
            self.idx2word.append(word)
            return index
        return self.word2idx[word]

    def get_id(self, word):
        if word not in self.word2idx:
            return -1
        else:
            return self.word2idx[word]

    @property
    def vocab_size(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, train_path, dev_path, test_path, feature='rnn', val=0.1):
        train_csv = pandas.read_csv(train_path)
        test_csv = pandas.read_csv(test_path)
        dev_csv = pandas.read_csv(dev_path)

        train_texts = train_csv['text'].tolist()
        dev_texts = dev_csv['text'].tolist()
        train_tags = train_csv['label'].tolist()
        dev_tags = dev_csv['label'].tolist()
        test_texts = test_csv['text'].tolist()

        train_len = len(train_tags)
        dev_len = len(dev_tags)

        # rand_idx = np.arange(len(train_texts))
        # np.random.shuffle(rand_idx)
        # train_val_idx = int(len(train_texts) * (1-val))
        # train_texts = [train_texts[i] for i in rand_idx]
        # train_tags = [train_tags[i] for i in rand_idx]

        count_vectorizer = CountVectorizer(min_df=1, lowercase=False, tokenizer=tokenizer)
        count = count_vectorizer.fit_transform(train_tags+dev_tags)
        count_array = count.toarray().astype('float32')
        self.train_targets = torch.from_numpy(count_array[:train_len])
        self.val_targets = torch.from_numpy(count_array[-dev_len:])
        self.id2tags = count_vectorizer.get_feature_names()

        if feature == 'rnn':
            # stop = nltk.corpus.stopwords.words('english')
            stop = stopwords
            self.dictionary = Dictionary()
            self.train_data = []
            for line in train_texts:
                tmp = []
                for word in line.split():
                    if word in stop:
                        continue
                    tmp.append(self.dictionary.add_vocab(word))
                self.train_data.append(tmp)
            self.val_data = self.train_data[train_len:]
            self.train_data = self.train_data[:train_len]

            self.test_data = []
            for line in test_texts:
                tmp = []
                for word in line.split():
                    ret = self.dictionary.get_id(word)
                    if ret == -1:
                        continue
                    tmp.append(ret)
                self.test_data.append(tmp)

        elif feature == 'tfidf':
            tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf = tfidf_vectorizer.fit_transform(
                train_texts + dev_texts + test_texts)
            tfidf_array = tfidf.toarray().astype('float32')

            self.train_data = torch.from_numpy(tfidf_array[:train_len])
            self.val_data = torch.from_numpy(tfidf_array[train_len:train_len + dev_len])
            self.test_data = torch.from_numpy(
                tfidf_array[train_len + dev_len:])

if __name__ == '__main__':
    """
    corpus = Corpus('data/', feature='tfidf')
    print(corpus.train_data)
    print(corpus.val_data)
    print(corpus.test_data)
    print(corpus.train_targets)
    print(corpus.val_targets)
    print(corpus.tags)
    """
    corpus = Corpus('data/', feature='rnn')
    print(len(corpus.train_data), corpus.train_data[0])
    print(len(corpus.val_data), corpus.val_data[0])
    print(len(corpus.test_data), corpus.test_data[0])
