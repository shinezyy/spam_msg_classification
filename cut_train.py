import jieba
from os.path import join as pjoin
from gensim.models import Word2Vec
from pprint import pprint
from array import array
import numpy as np
import pandas as pd
import time


start_time = time.time()
df = pd.read_csv(pjoin('data', 'labeled.txt'), sep='\t',
        header=None, na_filter=False, skip_blank_lines=False)
print('Loading takes {}s'.format(time.time() - start_time))

start_time = time.time()
data = df.values
sentences = []
for row in data:
    if row[-1] == '':
        sentence = ' '
    else:
        s = row[-1].replace(' ', '')
        sentence = ' '.join(jieba.cut(s))
    ss = [row[0], sentence]
    sentences.append(ss)
print('Cutting takes {}s'.format(time.time() - start_time))

string_df = pd.DataFrame(sentences, columns=['label', 'contents'])
string_df.to_csv(pjoin('data', 'cut-labeled.txt'),
        header=None, index=None, sep='\t')

