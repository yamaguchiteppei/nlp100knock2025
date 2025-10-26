def cipher(s):
    for i in range(len(s)):
        if s[i].islower():
            print(ord(s[i]))
        else:
            print(s[i])
    

cipher("ThankYou")
            