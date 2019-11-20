import numpy as np
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import pickle
import matplotlib.pyplot as plt
import collections
from rnn import *

#rnn、文章生成#データセット3パターン#テキスト、アルファベット列、数列

def get_pickle(filename):
    f = open(filename, "rb") #ファイルを読み込んで返す
    words = pickle.load(f)
    f.close()
    return words
def vocab_get(set): #vocabをファイルから読み込む
    if set==0: #テキスト
        vocab=get_pickle("train_txt_voc.pkl")
    if set==1: #アルファベット列
        vocab=get_pickle("train_alpha_voc.pkl")
    if set==2: #数列
        vocab = get_pickle("train_number_voc.pkl")
    return vocab

def predict(model,s,demb,id2word): #生成、単語を1つずつ入力する
    h=Variable(np.zeros((1,demb),dtype=np.float32))
    for i in range(len(s)):
        x_k=model.embed(Variable(np.array([s[i]],dtype=np.int32)))
        h=F.tanh(x_k+model.H(h))
        yv=F.softmax(model.W(h))
    result=np.argmax(yv.data)
    return id2word[result],result

def create_data(number,model):  # 生成実行
    for b in range(number):
        count = 0
        pred = None
        s = [random.randint(1, 10)] # スタート単語はランダム決定
        while pred != "<eos>":
            pred, result = predict(model, s, demb, id2word)  # pred:単語など,result:id
            print(pred, end=" ")
            s.append(result)
            count += 1
            if count >= 20: pred = "<eos>" # 長すぎたら止める
        print()
def create_set(set): #どのデータを生成するか選ぶ
    model_predict = MyRNN(len(vocab), demb)
    if set==0:name="train_txt"
    if set==1:name="train_alpha"
    if set==2:name="train_number"
    model_predict.embed = get_pickle(name+"_embed.pkl")
    model_predict.H = get_pickle(name+"_H.pkl")
    model_predict.W = get_pickle(name+"_W.pkl")
    return model_predict

set=0 #テキスト:0、アルファベット列:1、数列:2
vocab=vocab_get(set) #単語idの辞書
eos_id=vocab["<eos>"] #eosのid取得
demb=10 #分散表現の次元demb=k
id2word={} #idから単語に戻す辞書
for a,b in enumerate(vocab):id2word[a]=b

model_predict=create_set(set) #学習済みデータ読み込む
create_data(5,model_predict) #データ生成
