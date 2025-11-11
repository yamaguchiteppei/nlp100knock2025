N = 10
with open('popular-names.txt','r') as f:
    for i in range(N):
        line = f.readline()
        print(line)
