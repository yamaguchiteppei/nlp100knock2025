"""05. n-gram
与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ。この関数を用い、”I am an NLPer”という文から文字tri-gram、単語bi-gramを得よ。
"""

s = "I am an NLPer"

def str_n_gram(str,n):
    print([str[idx:idx+n] for idx in range(len(str)-n+1)])

def word_n_gram(str,n):
    words = str.split()
    print([words[idx:idx+n] for idx in range(len(words)-n+1)])

print("文字tri-gram:")
str_n_gram(s,3)

print("単語bi-gram:")
word_n_gram(s,2)