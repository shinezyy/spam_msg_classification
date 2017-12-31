import jieba
import numpy as np
import xgboost
from os.path import join as pjoin
from sklearn.feature_extraction.text import HashingVectorizer


def cut(msg: str) -> str:
    return ' '.join(jieba.cut(msg))


n_features = 100000
chinese_punctuations = [
    ' ', '，', '。', '？', '；', '"', '～', '【',
    '】', '/', '：', '『', '﹃', '﹄', '』', '《', '》',
    '──', '‘ ', '’', '“ ', '”', ' '
]


def new_hash_vectorizer():
    return HashingVectorizer(
        n_features=n_features,
        stop_words=chinese_punctuations,
    )


def vectorize(vectorizer, cut_msg: str) -> np.ndarray:
    return vectorizer.transform(np.array([cut_msg]))


def load_model(model_path: str):
    bst = xgboost.Booster({'nthread': 2})
    bst.load_model(model_path)
    return bst


def predict(vectorizer, model, msg: str) -> np.ndarray:
    cut_msg = cut(msg)
    vec = vectorize(vectorizer, cut_msg)
    dtest = xgboost.DMatrix(vec)
    ypred = model.predict(dtest)
    return ypred


def main():
    vectorizer = new_hash_vectorizer()
    model = load_model(pjoin('..', 'data', 'xgb.bin'))

    while True:
        msg = input('spam> ')
        y = predict(vectorizer, model=model, msg=msg)
        print('垃圾短信概率:{}'.format(y[0]))
        if y[0] > 0.5:
            print('是垃圾短信')
        else:
            print('不是垃圾短信')


if __name__ == '__main__':
    main()
