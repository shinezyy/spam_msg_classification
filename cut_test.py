import jieba
from os.path import join as pjoin
from gensim.models import Word2Vec
from pprint import pprint
from array import array
import numpy as np
import pandas as pd
import time


start_time = time.time()
df = pd.read_csv(pjoin('data', 'without-label.txt'), sep='\t',
        header=None, na_filter=False, skip_blank_lines=False)
print('Loading takes {}s'.format(time.time() - start_time))

start_time = time.time()
data = df.values
sentences = []
for row in data:
    if row[-1] == '':
        sentences.append(' ')
    else:
        s = row[-1].replace(' ', '')
        # s = s.replace('\r\n', '\r')
        sentences.append(' '.join(jieba.cut(s)))
print('Cutting takes {}s'.format(time.time() - start_time))

string_df = pd.DataFrame(sentences, columns=['contents'])
string_df.to_csv(pjoin('data', 'cut-no-label.txt'),
        header=None, index=None, sep='\t')

