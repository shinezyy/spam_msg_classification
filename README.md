### 目录结构

data: 存放数据
models: 存放模型
preproc：预处理、向量化


### 输入与输出的文件

为了避免中文文件名带来的问题，老师提供的文件被重命名为：

- labeled.txt = 带标签短信.txt
- without-label.txt = 不带标签短信.txt

下面依次介绍各个脚本的功能。

首先是preproc下面的脚本

- cut\_train.py：将labeled.txt作为输入，用jieba分词，输出cut-labeled.txt。
- cut\_test.py：将without-label.txt作为输入，用jieba分词，输出cut-no-label.txt。
- doc2vec.py：将cut-labeled.txt和cut-no-label作为输入，用word2vec算法，
分别输出习得的向量到vec-train.txt和vec-test.txt。
- labels.py：为了方便某些算法，将label列从cut-labeled.txt中剥离出来，输出到labels.txt
- to\_libsvm.py, to\_libsvm\_test.py：没用了
- tf\_vectorizer.py：用tfidf和hashing-vectorizing两种方法向量化句子，输出为
scipy的稀疏表示。cut-labeled.txt被向量化的结果为tfidf-vec-train.npz和
hash-vec-train.npz，cut-no-label.txt被向量化的结果为tfidf-vec-test.npz
和hash-vec-test.npz。

以上数据下载地址为
[Google Drive](https://drive.google.com/drive/folders/1xig7Zmk7cSK-z_VZ0oA-Pqt68aVHxu3r?usp=sharing "Shared by zyy")

接下来是models下面的脚本

- XBG.py：调用xgboost来进行预测
- boosting.py：调用sklearn的GBDT和AdaBoosting来预测
- svm.py：调用sklearn的SVM来预测（太慢了，我没跑完）
- records.txt：调用AdaBoosting和xgboost的一些结果

上面3个脚本依赖于以下4个文件：

- vec-train.txt
- tfidf-vec-train.npz
- hash-vec-train.npz
- labels.txt

它们都可以从
[Google Drive](https://drive.google.com/drive/folders/1xig7Zmk7cSK-z_VZ0oA-Pqt68aVHxu3r?usp=sharing "Shared by zyy")
下载

