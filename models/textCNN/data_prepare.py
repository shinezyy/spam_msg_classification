#encoding=utf-8
import jieba

def load_data_and_labels(data_file):
    examples = list(open(data_file, 'r').readlines())
    examples = [s.replace(" ",'') for s in examples]
    print("分词开始。。。")
    examples = [list(jieba.cut(s, cut_all=False)) for s in examples]
    print("分词结束。。。")
    train_X_pos = []
    train_Y_pos = []
    train_X_neg = []
    train_Y_neg = []
    for s in examples:
        #train_X.append(s[2:-1])
        if s[0] == '0':
            train_Y_neg.append([1,0])
            train_X_neg.append(s[2:-1])
        else:
            train_Y_pos.append([0,1])
            train_X_pos.append(s[2:-1])#去除标签、开头和结尾
    train_X_neg = [' '.join(s) for s in train_X_neg]
    train_X_pos = [' '.join(s) for s in train_X_pos]
    print("The number of positive:{}".format(len(train_X_pos)))
    print('The number of negotive:{}'.format(len(train_X_neg)))
    return [train_X_pos, train_Y_pos, train_X_neg, train_Y_neg]



