# %%
import re
import random

import requests

from collections import Counter
from matplotlib.pyplot import yscale, xscale, title, plot
import matplotlib.pyplot as plt

# %%
TEXT = requests.get('https://norvig.com/big.txt').text
len(TEXT)


# %%
def tokens(text):
    return re.findall(r'[a-z]+', text.lower())


# %%
WORDS = tokens(TEXT)
len(WORDS)

# %%
print(WORDS[:10])


# %%
def sample(bag, n=10):
    return ' '.join(random.choice(bag) for _ in range(n))


# %%
sample(WORDS)

# %%
COUNTS = Counter(WORDS)

# %%
print(COUNTS.most_common(10))

# %%
M = COUNTS['the']
yscale('log')
xscale('log')
title('Частота n-того наиболее часто встречаемого слова '
      'и линия 1/n')
plot([c for (w, c) in COUNTS.most_common()])
plot([M / i for i in range(1, len(COUNTS))])
plt.show()


# %%
def known(words):
    """Вернуть подмножество слов, которое есть в нашем словаре"""
    return {w for w in words if w in COUNTS}


def edits0(word):
    return {word}


def splits(word):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    pairs = splits(word)
    deletes = [a + b[1:] for (a, b) in pairs if b]
    transposes = [
        a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1
    ]
    replaces = [
        a + c + b[1:] for (a, b) in pairs for c in alphabet if b
    ]
    inserts = [a + c + b for (a, b) in pairs for c in alphabet]

    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


# %%
def correct(word):
    candidates = (
        known(edits0(word)) or
        known(edits1(word)) or
        known(edits2(word)) or
        [word]
    )

    return max(candidates, key=COUNTS.get)


# %%
splits('wird')

# %%
print(edits0('wird'))

# %%
print(edits1('wird'))

# %%
print(len(edits2('wird')))


# %%
def case_of(text):
    return (
        str.upper if text.isupper() else
        str.lower if text.islower() else
        str.title if text.istitle() else
        str
    )


def correct_match(match):
    word = match.group()
    return case_of(word)(correct(word.lower()))


def correct_text(text):
    return re.sub('[a-zA-Z]+', correct_match, text)


# %%
print(correct_text(input('Write the text:')))


# %%
def memo(f):
    cache = {}

    def fmemo(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    fmemo.cache = cache
    return fmemo


# %%
def splits(text, start=0, L=20):
    return [
        (text[:i], text[i:]) for i in range(
            start, min(len(text), L) + 1
        )
    ]


# %%
def pdist(counter):
    N = sum(list(counter.values()))
    return lambda x: counter[x] / N


P = pdist(COUNTS)


def products(nums):
    result = 1
    for x in nums:
        result *= x
    return result


def Pwords(words):
    return products(P(w) for w in words)


# %%
@memo
def segment(text):
    if not text:
        return []
    else:
        candidates = (
            [first] + segment(rest) for (first, rest) in splits(text, 1)
        )
        return max(candidates, key=Pwords)


# %%
print(' '.join(segment(input('Write the text:'))))

# %%

