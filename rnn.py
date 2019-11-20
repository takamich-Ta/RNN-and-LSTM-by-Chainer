import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import collections

class MyRNN(chainer.Chain): #rnn定義
    def __init__(self,v,k):
        super(MyRNN,self).__init__(
            embed=L.EmbedID(v,k),
            H=L.Linear(k,k), #ここをL.LSTMに変えるだけで使える
            W=L.Linear(k,v),
        )
    def __call__(self,s,eos_id):
        accum_loss=None
        v,k=self.embed.W.data.shape
        h=Variable(np.zeros((1,k),dtype=np.float32))
        for i in range(len(s)):
            next_w_id=eos_id if(i==len(s)-1) else s[i+1] #勾配の累積
            tx=Variable(np.array([next_w_id],dtype=np.int32))
            x_k=self.embed(Variable(np.array([s[i]],dtype=np.int32)))
            h=F.tanh(x_k+self.H(h))
            loss=F.softmax_cross_entropy(self.W(h),tx)
            accum_loss=loss if accum_loss is None else accum_loss+loss
        return accum_loss
