import pandas as pd
from random import *
import time

# 파일위치
filepath = 'Chatbot_data/quizfinal.csv'

class Englishteacher:

    global Qlen

    def __init__(self,filepath):
        self.filepath = filepath
        self.Question = pd.read_csv(self.filepath)
        self.Answer = pd.read_csv(self.filepath)
        time.sleep(1)
        print("헤헤헤헤")
        time.sleep(0.5)
        print("영어테스트를 시작한다능")
        

    def EnglishtQuestion(self):

        global Qlen
        Question_E = self.Question['Q']
        Qlen = randint(0,len(Question_E))

        return Question_E[Qlen]
    
    def EnglishAnswer(self):

        global Qlen
        Question_A = self.Question['A']

        return Question_A[Qlen]



def quizstart_v2():

    num = ['첫번째', '두번째', '세번째', '네번째', '다섯번째','여섯번째',
    '일곱번째', '여덟번째', '아홉번째', '열번째' ]

   
    falsecount = 0
    score = 0
    now = time.localtime()

        

    Et = Englishteacher(filepath)

    for i in range(10):
        time.sleep(0.5)
        print(num[i]+' 문제 라능')
        print(Et.EnglishtQuestion())
        time.sleep(0.5)
        for falsecount in range(2):
            time.sleep(0.5)
            Answer = input("잘보고 정답을 맞춰 보라능\n\
                    (그만하고 싶으면 '그만' 이라고 입력하라능): ")
            if Answer == Et.EnglishAnswer():
                print("와아 맞았다능 천재라능")
                score +=1
                break
            elif Answer == '그만':
                break
            elif falsecount < 1:
                print("바보라능 그것도 모른다능 다시 풀어보라능")
                falsecount += 1
                continue
            else:
                print(" 아까웠다능 답은 {} 이라능".format(Et.EnglishAnswer()))
                break
        if Answer == '그만':
            print('퀴즈는 여기까지 라능, 아쉽다능')
            break
    time.sleep(0.5)
    print("채점시간 이라능")
    time.sleep(1)
    print("%2d월%2d일 %02d시%02d분 오늘의 영어점수는 %d점 이라능"
    % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min,score*10))

      



