# 대화형태에서 답변시 어순의 형태를 분석해서 알려주기 위한 모델만들기

import pandas as pd
from konlpy.tag import Mecab
from konlpy.tag import Komoran
from sklearn.preprocessing import OneHotEncoder

Kmr = Komoran()

# pos naming

posbackup = ['일반명사', '고유명사', '의존명사', '대명사', '수사', '동사', '형용사', '보조용언', 
'긍정지정사', '부정지정사', '관형사', '일반부사', '접속부사', '감탄사', '주격조사', '보격조사', 
'관형격조사', '목적격조사', '부사격조사', '호격조사', '인용격조사', '보조사', '접속조사', 
'선어말어미', '종결어미', '연결어미', '명사형전성어미', '관형형전성어미', '체언접두사', \
'명사파생접미사', '동사파생접미사', '형용사파생접미사', '어근','기호','숫자','분석불능','외국어','물결','빗금','줄임표','화폐기호',\
'줄표']

posname = []

while 1:
    a = input('입력해주세요 ')
    if a=='end':
        break
    posname.append(a)

# pos naming

posename = []

i = 0
while 1:
    
    print('이번품사는 '+posname[i]+' 입니다')
    a = input('입력해주세요 ')
    i += 1
    if a=='end':
        break
    posename.append(a)



posename_backup = \
['NNG', 'NNP', 'NNB', 'NP', 'NR', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'JKS', 'JKC', 'JKG', 'JKO', \
'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR','SF','SN','NA','SL','SO',\
'SP','SE','SW','SS'] 


# Parts Of Speech Komoran dict 생성

Kmr_pos_dict = {}

for i in range(len(posename_backup)):
    Kmr_pos_dict[posename_backup[i]] = posbackup[i]

# 품사별 원핫인코딩

token = Kmr_pos_dict.keys()
token

wordindex = {}
for voca in token:
    if voca not in wordindex.keys():
        wordindex[voca] = len(wordindex)

# 인코딩 범위저장
one_hot_vector = [0]*(len(wordindex))

# 들어온 문서를 tokenizing 후 원핫인코딩으로 변환

def onehotencoder(text, wordindex):

    one_hot_vector = [0]*(len(wordindex))

    text = Kmr.pos(text)

    token = []
    for i in text:
        token.append(i[1])

    for word in token:
        index = wordindex[word]
        one_hot_vector[index] = 1
    return one_hot_vector

# 모델 train, test 데이터 구성을 위해 데이터 로드

total_train = pd.read_csv('Chatbot_data/total_train.csv',index_col=0)

total_train_Qincode = [] #Qdata set 변환

for i in total_train['Q']:
    total_train_Qincode.append(onehotencoder(i,wordindex))

total_train_Aincode = [] #Adata set 변환

for i in total_train['A']:
    total_train_Aincode.append(onehotencoder(i,wordindex))

import numpy as np
total_train_Qincode

a = np.array(total_train_Qincode)
b = np.array(total_train_Aincode)

np.savetxt('train_x_array.txt',a)
np.savetxt('train_y_array.txt',b)




# 인코딩으로 변환된 챗봇질의응답 데이터를 프레임으로 변환

wOrderModel_traindata = pd.DataFrame({'Q':a,'A':b})
wOrderModel_traindata.to_csv('Chatbot_data/wModel_traindata.csv')

# Komoran 오류 분석

text = total_train['Q'][22693]
text
Kmr.pos(text)
onehotencoder(text,wordindex)

len(total_train_Qincode)

total_train.iloc[22690:22695]







Mc = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
Kmr = Komoran()

text = '나는 학교에 간다'

Mc.pos(text)
Kmr.pos(text)

def tokenizer(list):
    
    tokenized_list = []

    for text in list:
        
        tokenized_list.append(Mc.pos(text))
        


import English_teacher as engt

engt.main()


