import random

def s_random(s):
    words1 = []
    words = s.split()
    for i in range(len(words)):
        if len(words[i]) > 4:
            chars = list(words[i])
            first = chars[0]
            middle = chars[1:len(chars)-2]
            last = chars[len(chars)-1]
            random.shuffle(middle)
            words1[i]= first + middle + last
        else:
            words1[i]= words[i]
    
    return words

s="I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind."
print(s_random(s))