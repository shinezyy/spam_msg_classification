import jieba
from os.path import join as pjoin
import pandas as pd
import time


def main():
    """
    This function cut labeled messages into words with jieba
    without-label.txt = 不带标签短信.txt, rename it to avoid encoding issues
    output to cut-no-label.txt
    """
    start_time = time.time()
    df = pd.read_csv(pjoin('..', 'data', 'without-label.txt'), sep='\t',
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
    string_df.to_csv(pjoin('..', 'data', 'cut-no-label.txt'),
                     header=None, index=None, sep='\t')


if __name__ == '__main__':
    main()
