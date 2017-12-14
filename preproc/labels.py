import pandas as pd
from os.path import join as pjoin


def main():
    """
    This script is used to extract labels from observations.
    Separated labels is convenient for validating on sparse matrices
    """
    labels = []
    with open(pjoin('..', 'data', 'cut-labeled.txt'), encoding='utf-8') as f:
        for line in f:
            label = line.split('\t')[0]
            labels.append(int(label))

    df = pd.DataFrame(labels, columns=['label'])
    df.to_csv(pjoin('..', 'data', 'labels.txt'),
              header=None, index=None, sep=' ')


if __name__ == '__main__':
    main()
