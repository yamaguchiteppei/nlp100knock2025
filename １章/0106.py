"""06. 集合
“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を、それぞれ, 
XとYとして求め、XとYの和集合（X∨Y）、積集合（X∧Y）、差集合（X\Y）を求めよ。さらに、’se’というbi-gramがXおよびYに含まれるかどうかを調べよ。
"""

s1 = "paraparaparadise"
s2 = "paragraph"

def bigram(s):
    return {s[i:i+2] for i in range(len(s)-1)}

X = bigram(s1)
Y = bigram(s2)

union = X | Y
intersect = X & Y
diff = X - Y

print("X:", X)
print("Y:", Y)
print("X ∨ Y:", union)
print("X ∧ Y:", intersect)
print("X - Y:", diff)
print("'se' in X:", "se" in X)
print("'se' in Y:", "se" in Y)
