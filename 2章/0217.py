#まず一列目を全て取り出す。
#一列目を取り出したら、それがリストなければ格納して、リストにあれば格納しないようにしてください。
list = []
with open('popular-names.txt','r') as f: 
    for line in f:
        ch = line.split()
        head = ch[0]
        if head not in list:
            list.append(head)

for word in list:
    print(word)