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
    for token in sent:
        if token.text =="メロス":
            head = token.head
            while head != head.head:
                if head.pos_ in["VERB","AUX"]:
                    print(f"主語:メロス→述語:{head.lemma_}")
                    break
                head = head.head
            if head.pos_ in ["VERB","AUX"]:
                print(f"主語:メロス→述語:{head.lemma_}")