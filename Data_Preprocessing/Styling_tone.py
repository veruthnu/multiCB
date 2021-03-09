from eunjeon import Mecab
from konlpy.tag import Komoran, Kkma
import hgtk
import pandas as pd
from tqdm.notebook import tqdm
komoran = Komoran()

def komoran_token_pos_flat_fn(string):
    tokens_ko = komoran.pos(string)
    pos = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
    return pos

def exchange_NP(target):
    keyword = []
    ko_sp = komoran_token_pos_flat_fn(target)
    for idx, word in enumerate(ko_sp):
        if word.find('NP') > 0:
            keyword.append(word.split('/'))
            break
    if keyword == []:
        return ''

    if keyword[0][0] == '저':
        keyword[0][0] = '나'
    elif keyword[0][0] == '제':
        keyword[0][0] = '내'
    else:
        return ''

    return keyword[0][0]

# 종결어미가 아닌 문장 중간의 말투가 변경되는 문제 해결을 위함
# 종결어미의 POS가 JX인 경우, EC가 맨 뒤에 위치하지 않는 경우를 제외
def non_JX(target):
    target = target.strip('.')
    ko_sp = komoran_token_pos_flat_fn(target)[::-1]
    for idx, word in enumerate(ko_sp):
        if (word.find('EC') > 0) & (idx == 0):
            pass
        else:
            target = target[::-1]
            if (word.find('JX') > 0) & (idx == 0):
                target = target.replace(word[0], '', 1)[::-1]
            else:
                target = target[::-1]
    return target

# 종결어미(EF, EC)만 저장
def make_special_word(target):
    ko_sp = komoran_token_pos_flat_fn(target)[::-1]
    keyword = []
    
    for idx, word in enumerate(ko_sp):
        if word.find('EF') > 0 :
            keyword.append(word.split('/'))
            _idx = idx
            break

        elif (word.find('EC') > 0) & (idx == 0):
            keyword.append(word.split('/'))
            _idx = idx
            break

        else:
            continue

    if keyword == []:
        return ''

    else:
        keyword = keyword[0]

    return keyword[0]

# ~능 말투를 만들어주기 위한 함수
def make_neung(target):
    target = target.rstrip(' ')
    target = target.rstrip(',')
    target = target.rstrip('.')
    hgtk_text = hgtk.text.decompose(target)

    if make_special_word(target) == 'ㄴ가요':
        if target.find('안가요') >= 0:
            target = target.replace(target[target.find('안가요'):], '안가냐능')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄱㅏᴥㅇㅛᴥ'):], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')
            target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄴ걸요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄱㅓㄹᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄴ다':
        target = target + '능'

    elif make_special_word(target) == 'ㄴ다고요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅏᴥㄱㅗᴥㅇㅛᴥ'):], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄴ다니':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅏᴥㄴㅣᴥ'):], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄴ다면':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅏᴥㅁㅕㄴᴥ'):], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄴ답니다':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂᴥㄴㅣᴥㄷㅏᴥ'):], 'ᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)
    
    elif make_special_word(target) == 'ㄴ데요':
        if target[-1] == '?':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅔᴥㅇㅛᴥ'):-1], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')

        elif hgtk_text[hgtk_text.find('ㄴᴥㄷㅔᴥㅇㅛᴥ')-1] == 'ㅏ':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅔᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')

        elif hgtk_text[hgtk_text.find('ㄴᴥㄷㅔᴥㅇㅛᴥ')-1] in ['ㅓ', 'ㅣ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅔᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) in ['ㄹ걸', 'ㄹ걸요']:
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㄱㅓㄹᴥ'):], 'ㄹᴥㄱㅓᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄹ게요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㄱㅔᴥㅇㅛᴥ'):], 'ㄹᴥㄱㅓᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄹ까요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㄲㅏᴥㅇㅛᴥ'):], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄹ래요':
        if target[-1] == '?':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㄹㅐᴥㅇㅛᴥ'):-1], 'ᴥㄱㅔㅆᴥㄴㅑᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㄹㅐᴥㅇㅛᴥ'):], 'ᴥㄱㅔㅆᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)
    
    elif make_special_word(target) == 'ㄹ지':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㅈㅣᴥ'):], 'ᴥㄱㅔㅆᴥㄴㅑᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㄹ텐데':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㅌㅔㄴᴥㄷㅔᴥ'):], 'ㄹᴥ ㄱㅓㅅᴥ ㄱㅏㅌᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) in ['ㅂ니다', 'ㅂ니다만']:
        if target[target.find(make_special_word(target))-1] in ['이', '르']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂᴥㄴㅣᴥㄷㅏᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂᴥㄴㅣᴥㄷㅏᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == 'ㅂ시다':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂᴥㅅㅣᴥㄷㅏᴥ'):], 'ᴥㅈㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '거나':
        target = target.replace(target[target.find('거나'):], '라능')

    elif make_special_word(target) == '거든요':
        if hgtk_text[hgtk_text.find('ᴥㄱㅓᴥㄷㅡㄴᴥㅇㅛᴥ')-1] == 'ㅆ':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄱㅓᴥㄷㅡㄴᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄱㅓᴥㄷㅡㄴᴥㅇㅛᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '게요':
        if target[-1] == '?':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄱㅔᴥㅇㅛᴥ'):-1], 'ㄹᴥㄱㅓᴥㄴㅑᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄱㅔᴥㅇㅛᴥ'):-1], 'ᴥㄱㅓᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '고요':
        if target[-1] == '?':
            target = target.replace(target[target.find('고요'):-1], '냐능')
        
        elif target[target.find(make_special_word(target))-1] in ['이', '하']:
            target = target.replace(target[target.find('고요'):], '라능')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄱㅗᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
            target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '군요':
        target = target.replace(target[target.find('군요'):], '다능')

    elif make_special_word(target) == '나요':
        if target[-1] == '?':
            target = target.replace(target[target.find('나요'):-1], '냐능')
        else:
            target = target.replace(target[target.find('나요'):], '냐능') + '?'

    elif make_special_word(target) == '네요':
        if target[target.find(make_special_word(target))-1] in ['이', '기']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅔᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        
        elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㅏ', 'ㅚ', 'ㅗ', 'ㅣ', 'ㅡ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅔᴥㅇㅛᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')

        elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㄷ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅔᴥㅇㅛᴥ'):], 'ㄴㅡㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        
        else: 
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅔᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '노라':
        target = target.replace(target[target.find('노라'):], '는다능')

    elif make_special_word(target) in ['는구나', '는군', '는군요']:
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅡㄴᴥㄱㅜㄴᴥ'):], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '는다면':
        target = target.replace(target[target.find('는다면'):], '냐능')

    elif make_special_word(target) == '는데요':
        if target[-1] == '?':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅡㄴᴥㄷㅔᴥㅇㅛᴥ'):-1], 'ᴥㄴㅑᴥㄴㅡㅇᴥ')

        elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㅚ', 'ㅏ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅡㄴᴥㄷㅔᴥㅇㅛᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅡㄴᴥㄷㅔᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '는지요':
        target = target.replace(target[target.find('는지요'):], '냐능')

    elif make_special_word(target) in ['니까', '니까요']:
        if target[target.find(make_special_word(target))-1] in ['르', '지', '가']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅣᴥㄲㅏ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif target[target.find(make_special_word(target))-1] == '하':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅣᴥㄲㅏ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif target[target.find(make_special_word(target))-1] == '테':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅌㅔᴥㄴㅣᴥㄲㅏ'):], 'ᴥㄱㅓᴥㄹㅏᴥㄴㅡㅇᴥ')
        elif target[target.find(make_special_word(target))-1] == '그':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅣᴥㄲㅏ'):], 'ᴥㄹㅓㅎᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅣᴥㄲㅏ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) in ['다', '다고']:
        target = target.replace(target[target.find('다'):], '다능')

    elif make_special_word(target) in ['다니', '다면']:
        target = target.replace(target[target.find('다'):], '냐능')

    elif make_special_word(target) == '답니다':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄷㅏㅂᴥㄴㅣᴥㄷㅏᴥ'):], 'ㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '대요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄷㅐᴥㅇㅛᴥ'):], 'ㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '더군요':
        if hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] == 'ㅚ':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄷㅓᴥㄱㅜㄴᴥㅇㅛᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄷㅓᴥㄱㅜㄴᴥㅇㅛᴥ'):], 'ㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) in ['더라고요', '더라구요']:
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄷㅓᴥㄹㅏᴥㄱ'):], 'ᴥㄷㅓᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '던가요':
        if target[-1] == '?':
            target = target.replace(target[target.find('던가요'):-1], '냐능')
        else:
            target = target.replace(target[target.find('던가요'):], '냐능')

    elif make_special_word(target) == '던데요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴᴥㄷㅔᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '데':
        target = target.replace(target[target.find('데'):], '다능')

    elif make_special_word(target) == '라네':
        target = target.replace(target[target.find('라네'):], '라능')

    elif make_special_word(target) in ['라니', '라니요']:
        target = target.replace(target[target.find('라니'):], '라능')

    elif make_special_word(target) == '라던데':
        target = target.replace(target[target.find('라던데'):], '라능')

    elif make_special_word(target) == '라면':
        target = target.replace(target[target.find('라면'):], '냐능')

    elif make_special_word(target) == '라서요':
        target = target.replace(target[target.find('라서요'):], '라능')

    elif make_special_word(target) == '랍니다':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂᴥㄴㅣᴥㄷㅏᴥ'):], 'ᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '래요':
        if target[target.find('래요')-1] == '저':
            target = target.replace(target[target.find('래요'):], '러냐능')
        elif target[target.find('래요')-1] == '뭐':
            target = target.replace(target[target.find('래요'):], '라는 거냐능')
        else:
            target = target.replace(target[target.find('래요'):], '냐능')

    elif make_special_word(target) == '려':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄹㅕᴥ'):], 'ᴥㄹㅣㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) in ['려고', '려고요']:
        if target[-1] == '?':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄹㅕᴥㄱㅗᴥ'):], 'ㄹᴥㄱㅓᴥㄴㅑᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄹㅕᴥㄱㅗᴥ'):], 'ㄹᴥㄱㅓᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '려나':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄹㅕᴥㄴㅏᴥ'):], 'ㄹᴥ ㄱㅓㅅᴥ ㄱㅏㅌᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '면':
        target = target.replace(target[target.find('면'):], '냐능')

    elif make_special_word(target) == '소':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅅㅗᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '습니까':
        target = target.replace(target[target.find('습니까'):], '냐능')

    elif make_special_word(target) == '습니다':
        target = target.replace(target[target.find('습니다'):], '다능')

    elif make_special_word(target) == '아서요':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅐᴥㅅㅓᴥㅇㅛᴥ'):], 'ㅏㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '아야죠':
        if target.find('해야죠') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅐᴥㅇㅑᴥㅈㅛᴥ'):], 'ㅏᴥㄹㅏᴥㄴㅡㅇᴥ')
        elif target[target.find('야죠')-1] == '아':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅇㅏᴥㅇㅑᴥㅈㅛᴥ'):], 'ᴥㅇㅏᴥㅇㅑᴥ ㅎㅏㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif target[target.find('야죠')-1] == '라':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅇㅑᴥㅈㅛᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅇㅑᴥㅈㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '아요':
        if target.find('????') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅏᴥㅇㅛᴥ'):], 'ㅡᴥㄴㅑᴥㄴㅡㅇᴥ')
        else:
            if hgtk_text.find('ㄱㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄱㅏᴥㅇㅛᴥ'):], 'ㄱㅏᴥㄴㅑᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㅎㅐᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅎㅐᴥㅇㅛᴥ'):], 'ㅎㅏㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㅌ','ㄹ', 'ㅆ', 'ㅎ', 'ㄶ', 'ㅄ', 'ㅈ']:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅇㅏᴥㅇㅛᴥ'):], 'ㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㄻ', 'ㄲ']:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅇㅏᴥㅇㅛᴥ'):], 'ㄴㅡㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] == 'ㅘ':
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅘᴥㅇㅛᴥ'):], 'ㅗᴥㄹㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㄴㅏᴥㅇㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴㅏᴥㅇㅏᴥㅇㅛᴥ'):], 'ㄴㅏㅅᴥㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ᴥㄴㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄴㅏᴥㅇㅛᴥ'):], 'ㄴㅏᴥㄴㅑᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ᴥㅂㅗᴥㅇㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂㅗᴥㅇㅏᴥㅇㅛᴥ'):], 'ㅂㅗᴥㄹㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㄹᴥㄹㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄹᴥㄹㅏᴥㅇㅛᴥ'):], 'ᴥㄹㅡᴥㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㅍㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅍㅏᴥㅇㅛᴥ'):], 'ㅍㅡᴥㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㅍㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅍㅏᴥㅇㅛᴥ'):], 'ㅍㅡᴥㄷㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㅁㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅁㅏᴥㅇㅛᴥ'):], 'ㅁㅏㄹᴥㄹㅏᴥㄴㅡㅇᴥ')
            elif hgtk_text.find('ㄷㅏㄹᴥㅇㅏᴥㅇㅛᴥ') > 0:
                hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄷㅏㄹᴥㅇㅏᴥㅇㅛᴥ'):], 'ㄷㅏㄹᴥㄷㅏᴥㄴㅡㅇᴥ')
            else:
                pass
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '어':
        if target.find('어') > 0:
            target = target.replace(target[target.find('어'):], '다능')
        else:
            pass

    elif make_special_word(target) == '어서요':
        if hgtk_text[hgtk_text.find('ᴥㅅㅓᴥㅇㅛᴥ')-1] == 'ㅝ':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅇㅝᴥㅅㅓᴥㅇㅛᴥ'):], 'ㅂᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif target[target.find('서요')-1] == '뻐':
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅃㅓᴥㅅㅓᴥㅇㅛᴥ'):], 'ㅃㅡᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅇㅓᴥㅅㅓᴥㅇㅛᴥ'):], 'ㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '어야죠':
        target = target.replace(target[target.find('야죠'):], '야 한다능')

    elif make_special_word(target) == '어야지요':
        target = target.replace(target[target.find('야지요'):], '야 한다능')

    elif make_special_word(target) == '어요':
        if target.find('세요') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅅㅔᴥㅇㅛᴥ'):], 'ㄹㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ㅆᴥㅇㅓᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅆᴥㅇㅓᴥㅇㅛᴥ'):], 'ㅆᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ㄷㅙᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㄷㅙᴥㅇㅛᴥ'):], 'ㄷㅚㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ᴥㅎㅐᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅎㅐᴥㅇㅛᴥ'):], 'ㅎㅏㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ᴥㅇㅝᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅇㅝᴥㅇㅛᴥ'):], 'ㅂᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ㅕᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅕᴥㅇㅛᴥ'):], 'ㅣㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅇㅓᴥㅇㅛᴥ'):], 'ㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '에요':
        target = target.replace(target[target.find('에요'):], '라능')

    elif make_special_word(target) == '예요':
        if target[-1] == '?':
            target = target.replace(target[target.find('예요'):-1], '냐능')
        else:
            target = target.replace(target[target.find('예요'):], '라능')

    elif make_special_word(target) == '요':
        if hgtk_text.find('ㅂㅏᴥㄹㅏᴥㅇㅛ') > 0: 
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅂㅏᴥㄹㅏᴥㅇㅛ'):], 'ㅂㅏᴥㄹㅏㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ᴥㅅㅔᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅅㅔᴥㅇㅛᴥ'):], 'ᴥㅅㅣᴥㄹㅏᴥㄴㅡㅇᴥ')
        elif hgtk_text.find('ᴥㄴㅏᴥㅇㅛᴥ') > 0:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㄴㅏᴥㅇㅛᴥ'):], 'ᴥㄴㅏㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '으니까요':
        target = target.replace(target[target.find('으니까요'):] ,'다능')

    elif make_special_word(target) == '으련만':
        target = target.replace(target[target.find('으련만'):] ,'겠다능')

    elif make_special_word(target) == '으면':
        target = target.replace(target[target.find('으면'):] ,'으면 좋겠다능')

    elif make_special_word(target) == '은가요':
        target = target.replace(target[target.find('은가요'):] ,'냐능')

    elif make_special_word(target) == '은데':
        target = target.replace(target[target.find('은데'):], '다능')

    elif make_special_word(target) == '은데요':
        target = target.replace(target[target.find('은데요'):], '다능')

    elif make_special_word(target) == '을걸요':
        target = target.replace(target[target.find('을걸요'):], '을 것 같다능')

    elif make_special_word(target) == '을까요':
        target = target.replace(target[target.find('을까요'):], '겠냐능')

    elif make_special_word(target) == '을지':
        target = target.replace(target[target.find('을지'):], '을지 모른다능')

    elif make_special_word(target) == '잖아요':
        if hgtk_text[hgtk_text.find('ᴥㅈㅏㄶᴥㅇㅏᴥㅇㅛᴥ')-1] in ['ㅆ', 'ㅎ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅏㄶᴥㅇㅏᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅏㄶᴥㅇㅏᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '죠':
        if hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㄹ', 'ㅆ', 'ㅎ', 'ㄶ', 'ㅄ']:
        #if hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㄹ', 'ㅆ', 'ㅏ', 'ㄶ', 'ㅄ', 'ㅅ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅈㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ㅈㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif make_special_word(target) == '지만':
        target = target.replace(target[target.find('지만'):], '다능')

    elif make_special_word(target) == '지요':
        if target[target.find(make_special_word(target))-1] in ['이']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅣᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㄱ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅣᴥㅇㅛᴥ'):], 'ᴥㄴㅡㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㅣ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅣᴥㅇㅛᴥ'):], 'ᴥㄹㅏᴥㄴㅡㅇᴥ')
        elif hgtk.text.decompose(target[target.find(make_special_word(target))-1])[-2] in ['ㅚ', 'ㅡ']:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅣᴥㅇㅛᴥ'):], 'ㄴᴥㄷㅏᴥㄴㅡㅇᴥ')
        else:
            hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅈㅣᴥㅇㅛᴥ'):], 'ᴥㄷㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif target[-1] == '예':
        target = target.replace(target[target.find('예'):], '라능')

    elif target[-1] == '세':
        hgtk_text = hgtk_text.replace(hgtk_text[hgtk_text.find('ᴥㅅㅔᴥ'):], 'ᴥㅅㅣᴥㄹㅏᴥㄴㅡㅇᴥ')
        target = hgtk.text.compose(hgtk_text)

    elif target[-1] == '네':
        target = target.replace(target[target.find('네'):], '라능')

    elif target[-1] == '거':
        target = target.replace(target[target.find('거'):], '거라능')

    else:
        pass

    return target

# 반말 딕셔너리
def get_rough_dic():
    my_exword = {
        '돌아와요': '돌아와',
        '으세요': '으셈',
        '잊어버려요': '잊어버려',
        '나온대요': '나온대',
        '될까요': '될까',
        '할텐데': '할텐데',
        '옵니다': '온다',
        '봅니다': '본다',
        '네요': '네',
        '된답니다': '된대',
        '데요': '데',
        '봐요': '봐',
        '부러워요': '부러워',
        '바랄게요': '바랄게',
        '지나갑니다': "지가간다",
        '이뻐요': "이뻐",
        '지요': "지",
        '사세요': "사라",
        '던가요': "던가",
        '모릅니다': "몰라",
        '은가요': "은가",
        '심해요': "심해",
        '몰라요': "몰라",
        '라요': "라",
        '더라고요': '더라고',
        '입니다': '이라고',
        '는다면요': '는다면',
        '멋져요': '멋져',
        '다면요': '다면',
        '다니': '다나',
        '져요': '져',
        '만드세요': '만들어',
        '야죠': '야지',
        '죠': '지',
        '해줄게요': '해줄게',
        '대요': '대',
        '돌아갑시다': '돌아가자',
        '해보여요': '해봐',
        '라뇨': '라니',
        '편합니다': '편해',
        '합시다': '하자',
        '드세요': '먹어',
        '아름다워요': '아름답네',
        '드립니다': '줄게',
        '받아들여요': '받아들여',
        '건가요': '간기',
        '쏟아진다': '쏟아지네',
        '슬퍼요': '슬퍼',
        '해서요': '해서',
        '다릅니다': '다르다',
        '니다': '니',
        '내려요': '내려',
        '마셔요': '마셔',
        '아세요': '아냐',
        '변해요': '뱐헤',
        '드려요': '드려',
        '아요': '아',
        '어서요': '어서',
        '뜁니다': '뛴다',
        '속상해요': '속상해',
        '래요': '래',
        '까요': '까',
        '어야죠': '어야지',
        '라니': '라니',
        '해집니다': '해진다',
        '으련만': '으련만',
        '지워져요': '지워져',
        '잘라요': '잘라',
        '고요': '고',
        '셔야죠': '셔야지',
        '다쳐요': '다쳐',
        '는구나': '는구만',
        '은데요': '은데',
        '일까요': '일까',
        '인가요': '인가',
        '아닐까요': '아닐까',
        '텐데요': '텐데',
        '할게요': '할게',
        '보입니다': '보이네',
        '에요': '야',
        '걸요': '걸',
        '한답니다': '한대',
        '을까요': '을까',
        '못해요': '못해',
        '베푸세요': '베풀어',
        '어때요': '어떄',
        '더라구요': '더라구',
        '노라': '노라',
        '반가워요': '반가워',
        '군요': '군',
        '만납시다': '만나자',
        '어떠세요': '어때',
        '달라져요': '달라져',
        '예뻐요': '예뻐',
        '됩니다': '된다',
        '봅시다': '보자',
        '한대요': '한대',
        '싸워요': '싸워',
        '와요': '와',
        '인데요': '인데',
        '야': '야',
        '줄게요': '줄게',
        '기에요': '기',
        '던데요': '던데',
        '걸까요': '걸까',
        '신가요': '신가',
        '어요': '어',
        '따져요': '따져',
        '갈게요': '갈게',
        '봐': '봐',
        '나요': '나',
        '니까요': '니까',
        '마요': '마',
        '씁니다': '쓴다',
        '집니다': '진다',
        '건데요': '건데',
        '지웁시다': '지우자',
        '바랍니다': '바래',
        '는데요': '는데',
        '으니까요': '으니까',
        '셔요': '셔',
        '네여': '네',
        '달라요': '달라',
        '거려요': '거려',
        '보여요': '보여',
        '겁니다': '껄',
        '다': '다',
        '그래요': '그래',
        '한가요': '한가',
        '잖아요': '잖아',
        '한데요': '한데',
        '우세요': '우셈',
        '해야죠': '해야지',
        '세요': '셈',
        '걸려요': '걸려',
        '텐데': '텐데',
        '어딘가': '어딘가',
        '요': '',
        '흘러갑니다': '흘러간다',
        '줘요': '줘',
        '편해요': '편해',
        '거예요': '거야',
        '예요': '야',
        '예' : '야',
        '습니다': '어',
        '아닌가요': '아닌가',
        '합니다': '한다',
        '사라집니다': '사라져',
        '드릴게요': '줄게',
        '다면': '다면',
        '그럴까요': '그럴까',
        '해요': '해',
        '답니다': '다',
        '주무세요': '자라',
        '마세요': '마라',
        '아픈가요': '아프냐',
        '그런가요': '그런가',
        '했잖아요': '했잖아',
        '버려요': '버려',
        '갑니다': '간다',
        '가요': '가',
        '라면요': '라면',
        '아야죠': '아야지',
        '살펴봐요': '살펴봐',
        '남겨요': '남겨',
        '내려놔요': '내려놔',
        '떨려요': '떨려',
        '랍니다': '란다',
        '돼요': '돼',
        '버텨요': '버텨',
        '만나': '만나',
        '일러요': '일러',
        '을게요': '을게',
        '갑시다': '가자',
        '나아요': '나아',
        '어려요': '어려',
        '온대요': '온대',
        '다고요': '다고',
        '할래요': '할래',
        '된대요': '된대',
        '어울려요': '어울려',
        '는군요': '는군',
        '볼까요': '볼까',
        '드릴까요': '줄까',
        '라던데요': '라던데',
        '올게요': '올게',
        '기뻐요': '기뻐',
        '아닙니다': '아냐',
        '둬요': '둬',
        '십니다': '십',
        '아파요': '아파',
        '생겨요': '생겨',
        '해줘요': '해줘',
        '로군요': '로군요',
        '시켜요': '시켜',
        '느껴져요': '느껴져',
        '가재요': '가재',
        '어 ': ' ',
        '느려요': '느려',
        '볼게요': '볼게',
        '쉬워요': '쉬워',
        '나빠요': '나빠',
        '불러줄게요': '불러줄게',
        '살쪄요': '살쪄',
        '봐야겠어요': '봐야겠어',
        '네': '네',
        '어': '어',
        '든지요': '든지',
        '드신다': '드심',
        '가져요': '가져',
        '할까요': '할까',
        '졸려요': '졸려',
        '그럴게요': '그럴게',
        '': '',
        '어린가': '어린가',
        '나와요': '나와',
        '빨라요': '빨라',
        '겠죠': '겠지',
        '졌어요': '졌어',
        '해봐요': '해봐',
        '게요': '게',
        '해드릴까요': '해줄까',
        '인걸요': '인걸',
        '했어요': '했어',
        '원해요': '원해',
        '는걸요': '는걸',
        '좋아합니다': '좋아해',
        '했으면': '했으면',
        '나갑니다': '나간다',
        '왔어요': '왔어',
        '해봅시다': '해보자',
        '물어봐요': '물어봐',
        '생겼어요': '생겼어',
        '해': '해',
        '다녀올게요': '다녀올게',
        '납시다': '나자'
    }
    return my_exword
