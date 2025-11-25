import spacy
from spacy import displacy

nlp = spacy.load("ja_ginza")

doc = nlp("メロスは激怒した。")

html = displacy.render(doc, style="dep", page=True)

with open("result.html", "w", encoding="utf-8") as f:
    f.write(html)
