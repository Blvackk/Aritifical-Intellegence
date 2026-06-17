import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re  #re --> Regular Expressions

paragraph= """The movie 3 Idiots is a powerful blend of humor, emotion,
and social commentary
9 that critiques India’s rigid education system while celebrating
friendship and
10 individuality. The story revolves around three engineering students—
11 Rancho, Farhan, and Raju—whose bond grows stronger as they navigate
12 the pressures of college life at Imperial College of Engineering.
13 Rancho, with his unconventional thinking, constantly challenges the
14 rote-learning methods enforced by the strict director Virus, encouraging
his friends
15 to pursue knowledge for its own sake rather than chasing grades.
16 Farhan, torn between his passion for wildlife photography and his
parents’
17 expectations, and Raju, struggling with poverty and fear of failure,
both find
18 courage through Rancho’s philosophy of “All is well.”
19 The film skillfully balances comedy—like Chatur’s hilarious speech
20 scene—with poignant moments, such as Joy Lobo’s tragic suicide,
21 which underscores the dangers of academic pressure.
22 Rancho’s romance with Pia, Virus’s daughter, adds warmth and humanity
23 to the narrative. In the climax, the revelation that Rancho is actually
24 Phunsukh Wangdu, a renowned scientist, reinforces the film’s central
25 message: true success comes from following one’s passion, innovating
26 with purpose, and living authentically. This timeless story resonates
27 with audiences worldwide, inspiring them to rethink the meaning of
education and success."""


# text Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r' ',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r' ',' ',text)


# Preparing the dataset
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in
stopwords.words('english')]


# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)
words = model.wv.key_to_index


# Finding Word Vectors
vector = model.wv['friendship']
similar = model.wv.most_similar('friendship')
similar = model.wv.most_similar('passion')
similar = model.wv.most_similar('humor')
similar = model.wv.most_similar('movie')


