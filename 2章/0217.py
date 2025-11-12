list = []
with open('popular-names.txt','r') as f: 
    for line in f:
        ch = line.split()
        head = ch[0]
        if head not in list:
            list.append(head)

for word in list:
    print(word)