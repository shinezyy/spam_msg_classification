import jieba
from os.path import join as pjoin
import pandas as pd
import time


def main():
    """
    This function cut labeled messages into words with jieba
    labeled.txt = 带标签短信.txt, rename it to avoid encoding issues
    output to cut-labeled.txt
    """
    start_time = time.time()
    df = pd.read_csv(pjoin('..', 'data', 'labeled.txt'), sep='\t',
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
    string_df.to_csv(pjoin('..', 'data', 'cut-labeled.txt'),
                     header=None, index=None, sep='\t')


if __name__ == '__main__':
    main()
