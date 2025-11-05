s = "Hi He Lied Because Boron Could Not Oxidize Flourine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

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