def get_third_column(line):
    """1行の文字列から3列目（数値）を取り出して返す関数"""
    parts = line.split()
    return int(parts[2])  

filename = "popular-names.txt"


with open(filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

sorted_lines = sorted(lines, key=get_third_column, reverse=True)

for line in sorted_lines:
    print(line)
