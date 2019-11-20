import pickle
import random

#アルファベットが順に並んだやつが入ったデータセット作る
#['a','b','c','d','<eos>','b','c','d','e','<eos>','k','l','m','n','o','p','<eos>']みたいな感じ

def data_create(count):
    words=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    result=[]
    for a in range(count):
        number=random.randint(0,10) #列の長さ
        start=random.randint(0,number) #スタート単語
        for a in range(number):result.append(words[a+start])
        result.append("<eos>")
    return result

a=data_create(100) #データ生成
f=open("train_alpha_raw.pkl","wb") #データ保存
pickle.dump(a,f)
f.close()
