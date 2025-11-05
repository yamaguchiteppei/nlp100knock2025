import random
s ="I couldn't believe that I could actually understand what I was reading: the phenomenal power of the human mind."

words = s.split()
new_words =[]
char = []
def  typoglycemia(words):
    for i in range(len(words)):
        if len(words[i]) > 4:
            first = words[i][0]
            last = words[i][len(words[i])-1]
            middle = list(words[i][1:len(words[i])-1])
            random.shuffle(middle)
            new_words.append(first+''.join(middle)+last)
        else:
            new_words.append(words[i])
    return " ".join(new_words)

print(typoglycemia(words))