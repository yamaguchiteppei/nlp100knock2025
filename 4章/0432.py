import spacy 

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

nlp = spacy.load('ja_ginza')
doc = nlp(text)

for sent in doc.sents:
    tokens = list(sent)
    for i in range(len(tokens)-1):
        if tokens[i].pos_ == "NOUN" and tokens[i+1].text == "の" and tokens[i+2].pos_ == "NOUN":
            print(f"{tokens[i].text}の{tokens[i+2].text}")