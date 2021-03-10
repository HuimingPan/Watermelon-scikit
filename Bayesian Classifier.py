import sklearn.naive_bayes as bayes
import numpy as np
from dataset import watermelon30 as wm3

def convert(wm3):
    attr_to_0=['否']
    attr_to_1=['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是']
    attr_to_2=['乌黑','稍蜷','沉闷','稍糊','稍凹','软粘']
    attr_to_3=['浅白','硬挺','清脆','模糊','平坦']
    for item in attr_to_0:
        np.place(wm3, wm3==item, int(0))
    for item in attr_to_1:
        np.place(wm3, wm3==item, int(1))
    for item in attr_to_2:
        np.place(wm3, wm3==item, int(2))
    for item in attr_to_3:
        np.place(wm3, wm3==item, int(3))
    wm3=np.array(list(map(lambda x:list(map(lambda y:np.double(y),x)),wm3)))
    return wm3
#    for rows in wm3:
      #  for element in rows:
def generate_dataset():
    dataset=wm3[:,1:-1]
    return dataset
def generate_labels():
    labels=wm3[:,-1]
    return labels

def evaluate(output,test_value):
    a=(output==test_value).all()
    b=0
    for item in a:
        if item==True:
            b+=1
    return b/len(output)
                
def onwatermelon3():
    '''
    西瓜书Page151，7.3贝叶斯分类器中
    用西瓜数据集3.0训练一个朴素贝叶斯分类器，并对编号为1的样本进行分类。
    -------
    None.

    '''
    #clf=bayes.GaussianNB()#[1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    #clf=bayes.MultinomialNB()#[1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    #clf=bayes.BernoulliNB()#[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #clf=bayes.CategoricalNB()#[1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0.]
    clf=bayes.ComplementNB()#[1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    clf.fit(dataset,labels)
    predict_value=clf.predict(dataset[:])
    print(predict_value)
    return clf

wm3=convert(wm3)
dataset=generate_dataset()
labels=generate_labels()
model=onwatermelon3()

    