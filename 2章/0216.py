import random

with open('popular-names.txt','r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    with open("random_shuffle.txt",'w') as out:
        out.writelines(lines)