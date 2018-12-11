from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.model_selection import cross_val_score

def random_search_multilabel(X,y):

    n_iter = 20
    scoring = 'f1_micro'
    verbose = 3
    n_jobs = -1
    random_state = 12
    cv = 4

    stop_words = {'he', 'don', 'very', 'of', 'shan', 'off', 'have', 'during', 'own', 'can', 'did', 'after', 'she', "didn't", 'their', 've', 'his', 'am', "should've", 'only', 'but', 'the', 'further', 'for', 'mightn', 'below', 'him', 'hers', 'or', 'such', "don't", 'my', 'those', 'some', 'as', 'down', 'before', 'which', 'an', 'couldn', 'aren', 'over', 'what', 'themselves', "aren't", 'your', 'where', 'himself', 'hasn', 'm', 'ours', 'we', 'more', 'ain', 'wouldn', 'and', 'weren', 'wasn', 'its', 'any', 'is', "mustn't", 'again', 'has', 'against', 't', 'other', 'each', 'out', 'doesn', 'they', 'isn', 'herself', 'than', "you've", "isn't", 'are', 'were', 'having', 'll', 'if', 'yourselves', 'myself', 'ma', 'through', 'our', 'same', 'just', 'being', 'o', 'no', "it's", 'hadn', 'itself', 'it', 'too', 'will', 'ourselves', 'from', 'all', 'when', 'why', "won't", "wouldn't", 'on', 'yours', 'needn', "doesn't", 'didn', "you're", 'at', "couldn't", 'had', 'how', 'nor', 'theirs', 'above', 'with', 'here', 'both', "mightn't", "you'd", 'do', 'into', 'because', 'be', 'under', 'who', 'whom', 'these', 'that', 'yourself', "needn't", 's', "weren't", 'once', 'does', 'then', 'about', 'until', 'in', 'you', 'not', 'this', 'mustn', "you'll", 're', 'should', 'haven', 'up', 'y', 'by', 'shouldn', 'd', "she's", "that'll", 'doing', 'there', "haven't", "shan't", 'been', "hasn't", 'while', 'won', 'now', "wasn't", 'a', 'i', 'her', 'so', "shouldn't", 'was', "hadn't", 'them', 'few', 'me', 'between', 'most', 'to'}

    a_vect = TfidfVectorizer()
    
    clf = LabelPowerset(classifier=RidgeClassifier(alpha=0.01, class_weight='balanced', copy_X=True,
                fit_intercept=False, max_iter=None, normalize=False,
                random_state=None, solver='auto', tol=0.001),
                require_dense=[True, True])

    pipeline = Pipeline([
        ('vect', a_vect), 
        ('clf', clf), 
    ])

    '''
    parameters = {'clf': [LabelPowerset(), BinaryRelevance(), ClassifierChain()],
                'clf__classifier': [RidgeClassifier()],
                'clf__classifier__alpha': [0.01,0.001, 0.0001],
                'clf__classifier__fit_intercept': [True, False],
                'clf__classifier__normalize': [True, False],
                'clf__classifier__class_weight': [None, 'balanced'],
                #'vect__stop_words': (None, stop_words),
                #'vect__strip_accents': ('ascii', None),
                #'vect__analyzer': ('word', 'char'),
                #'vect__ngram_range': ((1,1), (1,2), (1,3), (1,4), (1,5), (1,6)),
                #'vect__max_df': (0.5, 0.75, 1.0),
                #'vect__max_features': (None, 15000, 25000, 50000),
                #'vect__use_idf': (True, False),
                #'vect__smooth_idf': (True, False),
                #'vect__sublinear_tf': (True, False)
                }
    '''

    # NEW
    parameters = { 'vect__stop_words': (None, stop_words)
               #'vect__strip_accents': ('ascii', None),
               #'vect__analyzer': ('word', 'char'),
               #'vect__ngram_range': ((1,1), (1,2), (1,3), (1,4), (1,5), (1,6)),
               #'vect__max_df': (0.5, 0.75, 1.0),
               #'vect__max_features': (None, 15000, 25000, 50000),
               #'vect__use_idf': (True, False),
               #'vect__smooth_idf': (True, False),
               #'vect__sublinear_tf': (True, False)
            }

    rnd_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=n_iter,
        scoring=scoring, 
        n_jobs=n_jobs,
        cv=cv,
        refit=True,
        verbose=verbose,
        random_state=random_state
        )

    rnd_search.fit(X, y)

    print("* Best score: %f" % (rnd_search.best_score_))
    print("* Best params: %s" % (str(rnd_search.best_params_)))

    return rnd_search   

def reportRndsearch(rnd_search, X_train, y_train, X_test, y_test, classes):
    
    print("Results on the test_set (mails never seen by the model) : ")
    scores = cross_val_score(rnd_search.best_estimator_, X_test, y_test, cv=3)
    print ("Folds: %i std: %.2f" % (len(scores),np.std(scores)))
    print()
 
    y_pred = rnd_search.predict(X_test)
    print("- Metrics detail :")
    print(metrics.classification_report(y_test, y_pred, target_names=classes)) 
    
    print()
    print()
    print("- Best params : ")
    for k, v in rnd_search.best_params_.items():
        print("%s : %s" % (k, v))
    print()




