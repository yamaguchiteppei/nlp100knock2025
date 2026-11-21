with open("popular-names.txt", "r") as f:
    for i,line in enumerate(f):
        if i >= 10:
            break
        print(line.split("\t")[0])