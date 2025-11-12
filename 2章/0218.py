from collections import Counter

filename = "popular-names.txt"

with open(filename, "r") as f:
    first_col = [line.strip().split()[0] for line in f if line.strip()]


counter = Counter(first_col)

for name, count in counter.most_common():
    print(f"{name}\t{count}")
