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

"""実行結果
激怒→激怒
邪智→邪智
除か→除く
なら→なる
決意→決意
わから→わかる
ある→ある
吹き→吹く
遊ん→遊ぶ
暮し→暮す
来→来る
対し→対する
あっ→ある
"""