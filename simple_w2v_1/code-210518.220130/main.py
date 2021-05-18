# %%
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, treebank
from nltk.stem import PorterStemmer

import re

import numpy as np
from numpy.linalg import norm

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics._classification import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import gensim.downloader as api

# %%
nltk.download('punkt')

# %%
data = 'All work and no play makes jack a dull boy'

# %%
print(word_tokenize(data))

# %%
print(sent_tokenize(
    'I was going home. It was surprise!'
))

# %%
"""
Наибольший вклад в смысл предложения вносят слова, которые
ввстречаются не слишком часто и не слишком редко
"""

# %%
nltk.download('stopwords')  # слова, которые встречаются очень часто

# %%
stopWords = set(stopwords.words('english'))

# %%
len(stopWords)

# %%
print(stopWords)

# %%
res = [word for word in word_tokenize(data) if word not in stopWords]

# %%
# Пропал токен 'no'
print(res)

# %%
# Выделение корня слова
words = ['game', 'gaming', 'gamed', 'games', 'compacted']

# %%
# stemming не смотрит на контекст
ps = PorterStemmer()
list(map(ps.stem, words))

# %%
# Лемматизация (есть зависимомть от контекста)
raw = """
DENNIS: Listen, strange women lying in ponds distributing swords 
is no basis for a system of government. Supreme executive power 
derives from a mandate from the masses, not from some farcical 
aquatic ceremony.
"""
tokens = word_tokenize(raw)

# %%
nltk.download('wordnet')

# %%
wnl = nltk.WordNetLemmatizer()
print(list(map(wnl.lemmatize, tokens)))

# %%
# Можно указать часть речи слова
wnl.lemmatize('is', 'v')

# %%
# Расставление частей речи словам в предложении
nltk.download('averaged_perceptron_tagger')

# %%
sentences = nltk.sent_tokenize(data)
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

# %%
# Парсинг
nltk.download('treebank')

# %%
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()

# %%
# Регулярные выражения
# С помощью регулярных выражений можно искать,
# заменять и сентезировать строки по шаблонам
word = 'supercalifragilisticexpialidocious'
re.findall('[aeiou]|super', word)

# %%
re.findall('\d+', 'There is some numbers: 49 and 432')

# %%
re.sub('[,\.?!]', ' ', 'How, to? split. text!').split()

# %%
re.sub('[^A-z]', ' ', 'I 123 can 45 play 67 football').split()

# %%
# nlp = en_core_web_sm.load()

# %%
# doc = nlp(
#     u'Apple is looking at buying U.K. startup for 1$ billion'
# )

# %%
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

# %%
newsgroups_train = fetch_20newsgroups(subset='train')

# %%
list(newsgroups_train.target_names)

# %%
print(newsgroups_train.filenames.shape)

# %%
print(newsgroups_train.target.shape)

# %%
cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(
    subset='train', categories=cats
)

# %%
print(newsgroups_train.filenames.shape)

# %%
print(newsgroups_train.data[0])

# %%
print(newsgroups_train.target[:10])

# %%
# Векторизация с помощью TF-IDF
categories = [
    'alt.atheism', 'talk.religion.misc',
    'comp.graphics', 'sci.space'
]
newsgroups_train = fetch_20newsgroups(
    subset='train', categories=categories
)

# %%
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

# %%
vectorizer = TfidfVectorizer(lowercase=False)
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

# %%
vectorizer = TfidfVectorizer(min_df=0.2)
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

# %%
vectorizer = TfidfVectorizer(max_df=0.9)
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

# %%
vector = vectors.todense()[1]

# %%
print(vector)

# %%
print(vectors[vector != 0].shape)

# %%
stopWords = set(stopwords.words('english'))
wnl = nltk.WordNetLemmatizer()


# %%
def preproc1(text):
    return ' '.join([
        wnl.lemmatize(word) for word in word_tokenize(
            text.lower()
        ) if word not in stopWords
    ])


# %%
vectorizer = TfidfVectorizer(
    max_features=1500, preprocessor=preproc1
)
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)

# %%
# Косинусная мера между векторами
type(vectors)  # т.к. в этих векторах очень много нулей,
               # по умолчанию они записываются как sparce matrix

# %%
print(newsgroups_train.target[:10])

# %%
np.unique(newsgroups_train.target)

# %%
dense_vectors = vectors.todense()
print(dense_vectors.shape)


# %%
def cosine_sim(v1, v2):
    return np.array(v1 @ v2.T / norm(v1) / norm(v2))[0][0]


# %%
cosine_sim(dense_vectors[1], dense_vectors[1])

# %%
cosines = []
for i in range(10):
    cosines.append(cosine_sim(
        dense_vectors[0], dense_vectors[i]
    ))

# %%
# [1, 3, 2, 0, 2, 0, 2, 1, 2, 1]
print(cosines)  # самым близким оказался вектор из той же категории

# %%
svc = svm.SVC()

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dense_vectors, newsgroups_train.target, test_size=0.2
)

# %%
print(y_train.shape, y_test.shape)

# %%
svc.fit(X_train, y_train)

# %%
accuracy_score(y_test, svc.predict(X_test))

# %%
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
accuracy_score(y_test, sgd.predict(X_test))

# %%
# Классификация на основе embeddings
categories = [
    'alt.atheism', 'talk.religion.misc',
    'comp.graphics', 'sci.space'
]
newsgroups_train = fetch_20newsgroups(
    subset='train', categories=categories
)

# %%
# embeddings = api.load('glove-twitter-25')
embeddings = api.load('glove-twitter-100')

# %%
print(embeddings['fly'])


# %%
def vectorize_sum(comment):
    embedding_dim = embeddings.vectors.shape[1]
    features = np.zeros([embedding_dim], dtype='float32')

    words = preproc1(comment).split()
    for word in words:
        if word in embeddings:
            features += embeddings[f'{word}']

    return features


# %%
preproc1('I can swim').split()

# %%
vectorize_sum('I can swim')

# %%
X_wv = np.stack([
    vectorize_sum(text) for text in newsgroups_train.data
])
print(X_wv.shape)

# %%
X_train_wv, X_test_wv, y_train, y_test = train_test_split(
    X_wv, newsgroups_train.target, test_size=0.2
)

# %%
print(X_train_wv.shape, X_test_wv.shape)

# %%
wv_model = LogisticRegression().fit(X_train_wv, y_train)

# %%
accuracy_score(y_test, wv_model.predict(X_test_wv))
