s = "Hi He lied Because Boron Could Not Oxidize Flourine. New Nations Might Also Sign Peace Security Clause. Arther King Can"
s = s.replace('.','')
words = s.split()

one_char_idx = [1,5,6,7,8,9,15,16,19]
arr = {}

for i,word in enumerate(words,start=1):
    if i in one_char_idx:
        arr[i] = word[:1]
    else:
        arr[i] = word[:2]

dictionary  = {v:i for i,v in arr.items()}

print(dictionary)
