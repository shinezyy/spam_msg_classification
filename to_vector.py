from os.path import join as pjoin
from gensim.models import Word2Vec
from pprint import pprint
import pandas as pd
import time


start_time = time.time()
df = pd.read_csv(pjoin('data', 'cut-labeled.txt'), sep='\t',
        header=None, na_filter=False, skip_blank_lines=False)
print('Loading takes {}s'.format(time.time() - start_time))

start_time = time.time()
model = Word2Vec(df.values[:, 0], min_count=10, size=200)
print('Word2Vector takes {}s'.format(time.time() - start_time))
