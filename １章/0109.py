"""09. Typoglycemia
スペースで区切られた単語列に対して、各単語の先頭と末尾の文字は残し、それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ。
ただし、長さが４以下の単語は並び替えないこととする。
適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え、その実行結果を確認せよ。
"""

import random
s ="I couldn't believe that I could actually understand what I was reading: the phenomenal power of the human mind."

words = s.split()
new_words =[]
char = []

def  typoglycemia(words):
    for i in range(len(words)):
        if len(words[i]) > 4:
            first = words[i][0]
            last = words[i][-1]
            middle = list(words[i][1:-1])
            random.shuffle(middle)
            new_words.append(first+''.join(middle)+last)
        else:
            new_words.append(words[i])
    return " ".join(new_words)

print(typoglycemia(words))