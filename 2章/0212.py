N = 10

with open('popular-names.txt','r') as f:
    lines = f.readlines()
    for line in lines[-N:]:
        print(line)