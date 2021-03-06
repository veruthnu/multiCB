import pandas as pd
from konlpy.tag import Okt

# global memory = [] # 메인에서 선언해야함

def save_k_memory(q):
    memory.append(q)

from konlpy.tag import Kkma
import hgtk
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
kkma = Kkma()

# 1. 형태소 분석
def kkma_token_pos_flat_fn(string):
    kkma = Kkma()
    tokens_ko = kkma.pos(string)
    pos = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
    return pos

print(kkma_token_pos_flat_fn('그러게요. 어느덧 한달이에요.'))

for idx, word in enumerate(kkma_token_pos_flat_fn('그러게요. 어느덧 한달이에요.')):
    print(idx)
    if word.rfind('EF') > 0:
        print(word.split('/'))
        break


for idx, word in enumerate(kkma_token_pos_flat_fn('그러게요. 어느덧 한달이에요.')):
    print(idx)
    print(word)

# 딕셔너리 만들기

import pandas as pd
from konlpy.tag import Komoran
kmr = Komoran()


text = pd.read_csv('Chatbot_data/twit_data2.csv')
text = text['Q'][:10]
text

print(''.join(text))

kmr.

wordlist = kmr.nouns(''.join(text[:10]))
wordlist

wordcount = {}

wordcount.get('일')

for word in wordlist:
    wordcount[word] = wordcount.get(word,0) + 1
    key

wordcount