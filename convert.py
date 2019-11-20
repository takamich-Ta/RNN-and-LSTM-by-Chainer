import numpy as np
import pickle

#ファイルを読み込んでidのデータセット作る
#vocab:単語を受け取ってid渡す辞書

#テキストデータ読み込んでidデータセットとvocab辞書を返す
def load_data(filename):
    vocab={}
    #ばらばらに分ける、単語ごと、空白で
    words=open(filename).read().replace("\n","<eos>").strip().split()
    dataset=np.ndarray((len(words),),dtype=np.int32) #データが入る箱
    #新しい単語が出てきたら追加する,vocab:辞書、単語とid
    for i,word in enumerate(words):
        if word not in vocab:vocab[word]=len(vocab)
        dataset[i]=vocab[word]
    return dataset,vocab

#アルファベット列データ読み込んでidデータセットとvocab辞書を返す
def get_data(words):
    vocab={}
    dataset=np.ndarray((len(words),),dtype=np.int32)
    for i,word in enumerate(words):
        if word not in vocab:vocab[word]=len(vocab)
        dataset[i]=vocab[word]
    return dataset,vocab

#数列データ読み込んでidデータセットとvocab辞書を返す
def int_data(words):
    # 数列の単語id対応の辞書
    vocab = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, "<eos>": 10}
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i,word in enumerate(words):dataset[i] = vocab[word]
    return dataset,vocab

def get_pickle(filename):
    f = open(filename, "rb")
    words = pickle.load(f)
    f.close()
    return words
def put_pickle(filename,data):
    f=open(filename,"wb")
    pickle.dump(data,f)
    f.close()

#アルファベット列データ読み込んでidデータセットとid辞書ファイル保存
words=get_pickle("train_alpha_raw.pkl")
dataset,vocab=get_data(words)
put_pickle("train_alpha_con.pkl",dataset)
put_pickle("train_alpha_voc.pkl",vocab)
#数列データ読み込んでidデータセットとid辞書ファイル保存
words=get_pickle("train_number_raw.pkl")
dataset,vocab=get_data(words)
put_pickle("train_number_con.pkl",dataset)
put_pickle("train_number_voc.pkl",vocab)
#テキストデータ読み込んでidデータセットとid辞書ファイル保存
dataset,vocab=load_data("train.txt")
put_pickle("train_txt_con.pkl",dataset)
put_pickle("train_txt_voc.pkl",vocab)
