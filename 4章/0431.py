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
    for word in sent:
        if word.pos_ == "VERB":
            print(f"{word}→{word.lemma_}")