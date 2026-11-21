def cipher(s):
    result=""
    for ch in s:
        if ch.islower():
            result += chr(219-ord(ch))
        else:
            result += ch
    return result    

text = "Thank You"
encrypted =cipher(text)
print("暗号化:",encrypted)

decrypted = cipher(encrypted)
print("復号化:",decrypted)
            