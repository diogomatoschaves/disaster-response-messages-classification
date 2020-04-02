import pickle
import sys

import joblib
import nltk

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from models.verb_counter import VerbCounter
from utils.defaults import TABLE_NAME

classifier_params_defaults = {
    "RandomForestClassifier": {"n_estimators": 100, "max_depth": 50},
    "KNeighborsClassifier": {"n_neighbors": 10,},
}

grid_search_params_defaults = {
    "RandomForestClassifier": {
        "clf__estimator__n_estimators": [20, 50, 100, 200],
        "clf__estimator__max_depth": [2, 5, 10, 50, None],
        "clf__estimator__max_features": ['auto', 'log2'],
    },
    "KNeighborsClassifier": {"clf__estimator__n_neighbors": [5, 10, 25, 50]},
}

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language="english")

tokenize_with_numbers = lambda text: tokenize(text, True)
tokenize_with_numbers.__name__ = "with_numbers"
tokenize_without_numbers = lambda text: tokenize(text, False)
tokenize_without_numbers.__name__ = "without_numbers"


def download_nltk_packages(packages=None):
    if not packages:
        packages = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]

    nltk.download(packages, quiet=True)


def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(TABLE_NAME, engine)
    X = df["message"]
    y = df.iloc[:, 4:]

    return X, y, y.columns


def tokenize(text, include_numbers=True):

    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    text = re.sub(url_pattern, "urlplaceholder", text)

    pattern = r"[^a-zA-Z0-9]" if include_numbers else r"[^a-zA-Z]"

    text = re.sub(pattern, " ", text)

    pattern = r'[0-9]+'

    text = re.sub(pattern, 'numberplaceholder', text)

    words = word_tokenize(text)

    words = [
        stemmer.stem(lemmatizer.lemmatize(word.lower().strip()))
        for word in words
        if word not in stop_words
    ]

    return words


def build_model(
    clf_name=None, grid_search=False, classifier_params=None, grid_search_params=None
):

    if not classifier_params:
        classifier_params = {}

    if not grid_search_params:
        grid_search_params = {}

    try:
        if not clf_name:
            clf_name = "KNeighborsClassifier"
            classifier_params.update(classifier_params_defaults["KNeighborsClassifier"])

            clf = KNeighborsClassifier(**classifier_params)
        else:
            if len(classifier_params) == 0 and clf_name in classifier_params_defaults:
                classifier_params.update(classifier_params_defaults[clf_name])
            clf = eval(clf_name)(**classifier_params)
    except NameError:
        raise Exception(f"{clf_name} is not a valid Classifier")

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_processing', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('verb_counter_normalized', Pipeline([
                ('count_verbs', VerbCounter()),
                ('scaler', StandardScaler())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf, n_jobs=-1))
    ])

    if grid_search:
        param_grid = {
            "features__text_processing__vect__tokenizer": [None, tokenize_with_numbers, tokenize_without_numbers],
            "features__text_processing__tfidf__smooth_idf": [True, False],
        }

        if len(grid_search_params) == 0 and clf_name in grid_search_params_defaults:
            param_grid.update(grid_search_params_defaults[clf_name])
        else:
            param_grid.update(grid_search_params)

        pipeline = GridSearchCV(pipeline, param_grid=param_grid)

    return pipeline


def evaluate_model(model, X_test, y_test, category_names, plot=True):

    y_pred = model.predict(X_test)

    precisions = []
    recalls = []
    fscores = []
    accuracies = []
    actual_positives = []

    for i, column in enumerate(category_names):

        y_test_col = np.array(y_test)[:, i]
        y_pred_col = np.array(y_pred)[:, i]

        precision, recall, fscore, _ = precision_recall_fscore_support(y_test_col, y_pred_col, average='binary', zero_division=0)

        accuracy = accuracy_score(y_test_col, y_pred_col)

        actual_positive = np.bincount(y_test_col)

        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        accuracies.append(accuracy)
        actual_positives.append(actual_positive[1] if len(actual_positive) >= 2 else 0)

    results = pd.DataFrame({
        'category': category_names,
        'precision': precisions,
        'recall': recalls,
        'f_score': fscores,
        'accuracy': accuracies,
        'actual_positives': actual_positives
    })

    *precision_recall_fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro"
    )

    accuracy = accuracy_score(np.array(y_test), np.array(y_pred))

    results = results.append(
        pd.DataFrame([["total", *precision_recall_fscore, accuracy, 0]], columns=results.columns)
    )

    results.set_index("category", inplace=True)

    if plot:
        plot_results(results)

    return results


def plot_results(results):

    plt.figure(figsize=(13, 12))

    mask = np.zeros(results.shape)
    mask[:,-1] = True

    ax = sb.heatmap(
        results*100,
        vmin=results.values[:,:-1].ravel().min() * 100,
        vmax=results.values[:,:-1].ravel().max() * 100,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap='Blues',
        cbar_kws={'format': '%.0f%%'}
    );

    for (j,i), label in np.ndenumerate(results.values):
        if i == 4:
            ax.text(i+0.5, j+0.5, int(label),
                    fontdict=dict(ha='center',  va='center',
                                  color='black', fontsize=10))


def save_model(model, model_filepath):

    joblib.dump(model, model_filepath)


def main():

    download_nltk_packages()

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("Building model...")
        grid_search = False
        model = build_model(
            clf_name='RandomForestClassifier',
            grid_search=grid_search,
            classifier_params={'n_estimators': 200, 'n_jobs': -1}
        )

        print("Training model...")
        model.fit(X_train, y_train)

        if grid_search:
            print(model.best_params_)

        print("Evaluating model...")
        results = evaluate_model(model, X_test, y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

        plt.show()

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
