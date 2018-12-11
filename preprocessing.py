from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import re
import string
import hashlib
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from email.parser import Parser
from sklearn.feature_extraction.text import CountVectorizer

MULTILABEL_REPERTORY_NAME = "data/enron/multilabel/"

conversationPattern = re.compile("(?:.*Original Message.*(?:\n.*){1,7}?(?:\n.*From:.*)?(?:\n.*){1,7}?Subject:.*|(?:\n.*To:.*)?(?:\n.*){1,7}?Subject:.*|.*Forwarded by.*(?:\n.*){1,7}?(?:\n.*From:.*)?(?:\n.*){1,7}?Subject:.*)") 
mail_Pattern = re.compile("(?:.*Message.*(?:\n.*){1,7}?(?:\n.*From:.*)?(?:\n.*){1,7}?(?:\n.*X\-.*)?(?:\n.*X\-.*)?(?:\n.*X\-.*)?(?:\n.*X\-.*)?(?:\n.*X\-.*)?(?:\n.*X\-.*)?(?:\n.*X\-.*))")
stemmer = SnowballStemmer("english")

# PREPROCESS ENRON DATASET

def load_raw_dataset(PATH):
    data = []
    for folder in os.listdir(MULTILABEL_REPERTORY_NAME):
        for file in os.listdir(MULTILABEL_REPERTORY_NAME+folder):
            with open(os.path.join(MULTILABEL_REPERTORY_NAME+folder, file)) as f:
                content = f.read()
                data.append((folder, content))

    text_files = pd.DataFrame(data, columns=['Folder', 'Content'])
    text_files = pd.crosstab(text_files['Content'],text_files['Folder'])
    text_files.reset_index(inplace=True)

    return text_files

def get_clean_body(body):
    clean_list = re.split(conversationPattern, body)
    clean_list = [text.strip() for text in clean_list]
    return "\n".join(clean_list[:]) 

def get_clean_mailinfo(body):
    clean_list = re.split(mail_Pattern, body)
    clean_list = [text.strip() for text in clean_list]
    return "\n".join(clean_list[:]) 

def clean_text(text):
    text = text.lower()
    words_list = [w for w in re.findall(r"[\w']+", text)]
    text = ' '.join(stemmer.stem(w) for w in words_list)
    return text

def remove_links(text):
    return re.sub(r'http\S+', '', text)

def collapse_whitespace(text): 
    return ' '.join(text.split())

def remove_all_num(text):
    for x in text:
        if x.isdigit():
            text = text.replace(x,'')
    return text

def remove_all_emails(text):
    text = re.sub('\S*@\S*\s?','', text)
    return text

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
