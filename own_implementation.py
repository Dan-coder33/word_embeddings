# %%
from main import preproc1

import re
from collections import Counter

import requests
from tqdm.notebook import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from catalyst import dl

# %% Practice
# Variant 1
"""
class Model(nn.Module):
    def __init__(
    self, 
    voc_size - величина словаря (сколько векторов храниться в embaddinge), 
    emb_dim - ширина матрицы
    ):
        self.u = nn.Embedding(voc_size, emb_dim)  
        # отличие nn.Embedding от linear - 
        на вход получает какой-то индекс, и выдает столбец, 
        либо строку этой матрицы
        self.v = nn.Embedding(voc_size, emb_dim)

w2v = Model(...)

def step(word, context):
    for c_word in context:
        loss = - w2v.u(word).T.dot(w2v.v(c_word)) - векторно умножаем центральное слово на контекстное
        cum_exp = 0
        for i in range(voc_size): - бежим по всему словарю
            if i == c_word:
                continue
            cum_exp += w2v.u(word).T.dot(w2v.v(c_word)).exp() - считаем сумму экспонент
        loss += torch.log(cum_exp) - берем логариф от суммы и прибавим к loss
        loss.backward()
        ...
"""

# Variant 2 - cводим нашу задачу к задаче классификации
"""
class Model(nn.Module):
    def __init__(self, voc_size, emb_dim):
        self.u = nn.Embedding(voc_size, emb_dim)
        self.v = nn.Linear(emb_dim, voc_size, bias=False)

    def forward(self, x):
        return self.v(self.u(x))

w2v = Model(...)
criterion = nn.CrossEntropyLoss() - как раз та loss функция, которая нам нужна

def step(word, context):
    for c_word in context:
        preds = w2v(word) - засовываем слово в нашу модель, модель выдает вероятности для каждого слова, 
        которое может встретиться в контексте
        loss = criterion(preds, c_word) - сравниваем то, что получилось у модели с тем, что было на самом деле
        loss.backward()
        ...
"""


# %%
class W2VCorpus:
    def __init__(
            self, path, voc_max_size: int = 40000,
            min_word_freq: int = 20, max_corp_size=5e6
    ):
        corpus = []
        sentences = []
        with open(path, "r") as inp:
            for line in inp:
                corpus.append(line.split())
                sentences.append(line)
        corpus = np.array(corpus)
        self.corpus = corpus
        most_freq_word = \
            Counter(' '.join(sentences).split()).most_common(voc_max_size)
        most_freq_word = np.array(most_freq_word)
        most_freq_word = \
            most_freq_word[most_freq_word[:, 1].astype(int) > min_word_freq]

        print('Vocabulary size is:' + str(len(most_freq_word)))
        self.vocabulary = set(most_freq_word[:, 0])
        self.vocabulary.update(["<PAD>"])
        self.vocabulary.update(["<UNK>"])
        self.word_freq = most_freq_word
        self.idx_to_word = dict(list(enumerate(self.vocabulary)))
        self.word_to_idx = \
            dict([(i[1], i[0]) for i in enumerate(self.vocabulary)])
        self.W = None
        self.P = None
        self.positive_pairs = None

    def make_positive_dataset(self, window_size=2):
        """take corpus and make positive examples for skipgram or CBOW
           like: [1234], [[3333, 1111, 2222, 4444]]"""
        if self.W is not None:
            return self.W, self.P
        W = []
        P = []
        pbar = tqdm(self.corpus)
        pbar.set_description('Creating context dataset')
        for message in pbar:

            if len(self.corpus) == 1:
                iter_ = tqdm(enumerate(message), total=len(message))
            else:
                iter_ = enumerate(message)

            for idx, word in iter_:
                if word not in self.vocabulary:
                    word = "<UNK>"
                start_idx = max(0, idx - window_size)
                end_idx = min(len(message), idx + window_size + 1)
                pos_in_window = window_size
                if idx - window_size < 0:  # start of the sentence
                    pos_in_window += idx - window_size

                co_words = message[start_idx:end_idx]  # cuts window from sentence
                co_words = np.delete(co_words, pos_in_window)  # deletes central word from context
                filtered_co_words = []

                for co_word in co_words:
                    if co_word in self.vocabulary:
                        filtered_co_words.append(co_word)
                    else:
                        filtered_co_words.append("<UNK>")
                while len(filtered_co_words) < 2 * window_size:
                    filtered_co_words.append("<PAD>")
                W.append(self.word_to_idx[word])
                co_word_idx = [self.word_to_idx[co_word] for co_word in filtered_co_words]
                P.append(co_word_idx)
        self.W = W
        self.P = P
        del self.corpus
        return W, P

    def make_positive_pairs(self):
        """[1234], [[3333, 1111, 2222, 4444]] ->
            [1234, 3333],
            [1234, 1111],
            ....
            [9999, 1982],
        """
        if self.positive_pairs is not None:
            return self.positive_pairs
        if self.W is None:
            self.make_positive_dataset()
        pairs = []
        pbar = tqdm(zip(self.W, self.P), total=len(self.W))
        pbar.set_description('Creating positive pairs')
        for w, p in pbar:
            for cur_p in p:
                if cur_p != self.word_to_idx["<PAD>"]:  # pad
                    pairs.append([w, cur_p])
        self.positive_pairs = pairs
        return pairs


# %%
with open('data/text', 'w') as f:
    f.write(
        preproc1(requests.get('https://norvig.com/big.txt').text)
    )
corp = W2VCorpus('data/text')
pairs = corp.make_positive_pairs()


# %%
class W2VDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, idx):
        return {
            "word": torch.tensor(self.pairs[idx][0]),
            "context": torch.tensor(self.pairs[idx][1])
        }

    def __len__(self):
        return len(self.pairs)


# %%
train_ds = W2VDataset(pairs)
train_dl = DataLoader(train_ds, batch_size=2048)
loaders = {"train": train_dl}


# %%
class W2VModel(nn.Module):
    def __init__(self, voc_size, emb_dim):
        super().__init__()
        self.encoder = nn.Embedding(voc_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, voc_size, bias=False)
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.init_emb()

    def forward(self, word):
        return self.decoder(self.encoder(word))

    def init_emb(self):
        """
        init the weight as original word2vec do.
        """
        initrange = 0.5 / self.emb_dim
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(0, 0)


# %%
model = W2VModel(len(corp.vocabulary), 300)
runner = dl.SupervisedRunner(
    input_key=["word"],
    input_target_key=["context"]
)

# %%
runner.train(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loaders=loaders,
    criterion=nn.CrossEntropyLoss(),
    callbacks=[dl.CriterionCallback(
        input_key='context'
    )],
    num_epochs=1,
    logdir="simple_w2v_1",
    verbose=False
)

# %%
# Default settings
"""
Метод: Skipgram

Размер Negative sample: about 5 for big dataset, 10-20 for small

Embedding space dim: 300 (the quality remains the same for higher dims)

Размер окна: 5-10
"""
# in case you want to try to create embeddings yourself

all_embeddings = []
all_words = []

for word, idx in corp.word_to_idx.items():
    with torch.no_grad():
        current_emb = model.encoder(torch.tensor(idx))
        current_emb = current_emb.cpu().detach().numpy()
        all_embeddings.append(current_emb)
        all_words.append(word)
all_embeddings = np.array(all_embeddings)
all_words = np.array(all_words).astype(str)
np.savetxt(
    'embeddings_t8.tsv',
    all_embeddings[:5000],
    delimiter='\t'
)

with open('words_t8.tsv', 'w') as out:
    for word in all_words[:5000]:
        out.write(word + '\n')
