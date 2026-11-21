n = int(input("nを入力してください:"))

with open("popular-names.txt", "r") as f:
    for i,line in enumerate(f):
        if i >= n:
            break
        print(line.strip())
        