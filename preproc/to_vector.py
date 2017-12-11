from os.path import join as pjoin
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import time


def main():
    """
    This function convert words into vectors with doc2vec
    cut-labeled.txt is labeled sentenced produced by cut_train.py
    cut-no-label.txt is sentences without label produced by cut_test.py
    output to vec-train.txt and vec-test.txt
    """
    start_time = time.time()
    with open(pjoin('..', 'data', 'cut-labeled.txt')) as f:
        matrix = f.read().split('\n')[:-1]
    print('train len: {}'.format(len(matrix)))
    print('Loading takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    sentences = []
    for i in range(len(matrix)):
        row = matrix[i].split('\t')
        sentence = TaggedDocument(row[-1].split(' '), [str(i)])
        sentences.append(sentence)

    model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=2,)
    print('Doc2Vector training takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    vectoriezed = []
    for row in matrix:
        row = row.split('\t')
        vec = list(model.infer_vector(row[-1]))
        vec.append(int(row[0]))
        vectoriezed.append(vec)

    df_out = pd.DataFrame(vectoriezed)
    out_file = pjoin('..', 'data', 'vec-train.txt')
    df_out.to_csv(out_file, header=None, index=None, sep=',')
    print('write to {}'.format(out_file))
    print('vectorizing train set takes {}s'.format(time.time() - start_time))

    start_time = time.time()
    with open(pjoin('..', 'data', 'cut-no-label.txt')) as f:
        matrix = f.read().split('\n')[:-1]
    print('test len: {}'.format(len(matrix)))

    vectoriezed = []
    for row in matrix:
        vec = list(model.infer_vector(row))
        vectoriezed.append(vec)

    df_out = pd.DataFrame(vectoriezed)
    out_file = pjoin('..', 'data', 'vec-test.txt')
    df_out.to_csv(out_file, header=None, index=None, sep=',')
    print('write to {}'.format(out_file))
    print('vectorizing test set takes {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
