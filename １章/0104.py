"""“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”
という文を単語に分解し、
1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字、
それ以外の単語は先頭の2文字を取り出し、取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）
を作成せよ。"""

s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
s = s.replace(".","").replace(",","")
words = s.split()
h1_idx = [1,5,6,7,8,9,15,16,19]
h2_idx = [2,3,4,10,11,12,13,14,17,18,20]
dictionary ={}

for i,word in enumerate(words,start=1):
    if i in h1_idx:
      dictionary[word[:1]] = i
    elif i in h2_idx:
      dictionary[word[:2]] = i

print(dictionary)