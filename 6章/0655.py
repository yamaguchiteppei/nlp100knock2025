import pandas as pd

df = pd.read_csv("ans54.txt", sep=" ", header=None)
print((df[3] == df[4]).sum() / len(df))

#出力結果(意味的アナロジー)しか求められなかった。
#0.8320158102766798