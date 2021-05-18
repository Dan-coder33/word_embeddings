# %%
import requests
from nltk .tokenize import WordPunctTokenizer
from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook

# %%
quora = requests.get(
    'https://www.dropbox.com/s/obaitrix9jyu84r/quora.txt?dl=1'
).text.split('\n')

# %%
tokenizer = WordPunctTokenizer()

# %%
print(tokenizer.tokenize(quora[13]))

# %%
quora_tokenized = [
    tokenizer.tokenize(line.lower()) for line in quora
]

# %%
model = Word2Vec(
    quora_tokenized, vector_size=32, min_count=5, window=3
).wv

# %%
model.get_vector('tv')

# %%
model.most_similar('tv')

# %%
n_words = 1000
words = sorted(
    list(model.index_to_key),
    key=lambda word: model.key_to_index[word]
)[:n_words]
print(words[::10])

# %%
word_vectors = [model.get_vector(word) for word in words]

# %%
pca = PCA(n_components=2)
pca.fit(word_vectors)
word_vectors_pca = pca.transform(word_vectors)

# %%
ss = StandardScaler().fit(word_vectors_pca)
word_vectors_pca = ss.transform(word_vectors_pca)

# %%
output_notebook()


def draw_vectors(
        x, y, radius=10,
        alpha=0.25, color='blue',
        width=600, height=400, show=True, **kwargs
):
    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource(
        {'x': x, 'y': y, 'color': color, **kwargs}
    )

    fig = pl.figure(
        active_scroll='wheel_zoom',
        width=width,
        height=height
    )

    fig.scatter(
        'x', 'y', size=radius,
        color='color', alpha='alpha',
        source=data_source
    )

    fig.add_tools(
        bm.HoverTool(tooltips=[
            (key, '@' + key) for key in kwargs.keys()
        ])
    )

    if show:
        pl.show(fig)

    return fig


# %%
draw_vectors(
    word_vectors_pca[:, 0],
    word_vectors_pca[:, -1],
    token=words
)

# %%

