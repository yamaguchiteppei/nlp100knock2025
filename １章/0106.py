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
print("X ∪ Y:", union)
print("X ∩ Y:", intersect)
print("X - Y:", diff)
print("'se' in X:", "se" in X)
print("'se' in Y:", "se" in Y)
