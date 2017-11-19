from os.path import join as pjoin
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from pprint import pprint
import pandas as pd
import time


start_time = time.time()
df = pd.read_csv(pjoin('data', 'cut-labeled.txt'), sep='\t',
        header=None, na_filter=False, skip_blank_lines=False)
matrix = df.values[:1000]
print('Loading takes {}s'.format(time.time() - start_time))

start_time = time.time()
sentences = []
for i in range(len(matrix)):
    row = matrix[i]
    sentence = LabeledSentence(row[0].split(' '), [str(i)])
    sentences.append(sentence)

model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, \
        epochs=2,
        )
print('Doc2Vector takes {}s'.format(time.time() - start_time))

print(matrix[0][0])
print(model.infer_vector(matrix[0][0]))
