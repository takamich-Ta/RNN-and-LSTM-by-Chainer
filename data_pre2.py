import random
import pickle

#フィボナッチ数列の一桁のみver、二桁になったら1桁のみ保持
#列数は10固定
#[2,1,3,4,7,1,8,9,7,6,3,9,"<eos>"]みたいな感じ

def data_create2(count):
    result=[]
    for c in range(count):
        data=[]
        a=random.randint(0,9) #最初の2数字ランダム生成
        b=random.randint(0,9)
        data.append(a)
        data.append(b)
        count=10
        for a in range(count):
            f=data[a]+data[a+1] #足す
            if f>=10:f=f-10
            data.append(f)
        data.append("<eos>")
        result=result+data
    return result

a=data_create2(100) #データ生成
f=open("train_number_raw.pkl","wb") #データ保存
pickle.dump(a,f)
f.close()
