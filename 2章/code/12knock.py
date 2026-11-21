n = int(input("nを入力してください:"))

with open("popular-names.txt", "r") as f:
    lines = f.readlines()
    for line in lines[-n:]:
        print(line.strip())