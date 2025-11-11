with open('popular-names.txt','r') as f:
    for i in range(10):
        line = f.readline()
        if not line:
            break
        line = line.replace('\t',' ')
        print(line)