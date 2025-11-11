with open('popular-names.txt','r') as f:
    for i in range(10):
        line = f.readline()
        line = line.strip().split()
        print(line[0])