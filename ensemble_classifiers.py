from skmultilearn.problem_transform import ClassifierChain, LabelPowerset, BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
import skmultilearn
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN

n_iter = 20
scoring = 'f1_micro'
verbose = 3
n_jobs = 4
random_state = 12
cv = 4

def build_Rake(X_train, y_train, X_test, y_test):

        classifier = RakelD(
            base_classifier=GaussianNB(),
            base_classifier_require_dense=[True, True],
            labelset_size=4
        )

        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))

def build_MajorityVoting(X_train, y_train, X_test, y_test):

        classifier = MajorityVotingClassifier(
            clusterer = FixedLabelSpaceClusterer(clusters = [[1,2,3], [0, 2, 5], [4, 5]]),
            classifier = ClassifierChain(classifier=GaussianNB())
        )
        classifier.fit(X_train,y_train)
        prediction = classifier.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))

def build_Mklnn(X_train, y_train):
    
        parameters = {
            'classifier': [LabelPowerset(), ClassifierChain()],
            'classifier__classifier': [RandomForestClassifier()],
            'classifier__classifier__n_estimators': [10, 20, 50],
        }

        clf = GridSearchCV(LabelSpacePartitioningClassifier(), parameters, scoring = 'f1_macro')
        clf.fit(X_train, y_train)

        print (clf.best_params_, clf.best_score_)




