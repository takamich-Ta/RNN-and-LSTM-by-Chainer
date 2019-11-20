import numpy as np
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import pickle
import collections
from rnn import *

#学習して、パラメータをファイルに保存する

def get_pickle(filename):
    f = open(filename, "rb") #ファイルを読み込んで返す
    words = pickle.load(f)
    f.close()
    return words
def put_pickle(filename,data):
    f=open(filename,"wb") #ファイルに保存する
    pickle.dump(data,f)
    f.close()

def traindata_vocab_get(set): #traindataとvocabをファイルから読み込む
    if set==0: #テキスト
        train_data=get_pickle("train_txt_con.pkl")
        vocab=get_pickle("train_txt_voc.pkl")
    if set==1: #アルファベット列
        train_data=get_pickle("train_alpha_con.pkl")
        vocab=get_pickle("train_alpha_voc.pkl")
    if set==2: #数列
        train_data = get_pickle("train_number_con.pkl")
        vocab = get_pickle("train_number_voc.pkl")
    return train_data,vocab

def train(number): #学習 #number:学習回数
    for epoch in range(number):
        s=[]
        for pos in range(len(train_data)):
            id=train_data[pos]
            s.append(id)
            if(id==eos_id):
                model.cleargrads()
                loss=model(s,eos_id)
                loss.backward()
                optimizer.update()
                s=[]

def train_put(set,model): #trainの結果をファイルに保存する
    if set==0:name="train_txt"
    if set==1:name="train_alpha"
    if set==2:name="train_number"
    put_pickle(name+"_embed.pkl", model.embed)
    put_pickle(name+"_H.pkl", model.H)
    put_pickle(name+"_W.pkl", model.W)

set=2 #txt読み込みが0、データ生成の英単語列が1、データ生成の数列が2
train_data,vocab=traindata_vocab_get(set)

eos_id=vocab["<eos>"] #eosのid取得

demb=10 #分散表現の次元demb=k
model=MyRNN(len(vocab),demb)
optimizer=optimizers.Adam()
optimizer.setup(model)

train(100) #学習実行
train_put(set,model) #学習結果保存
