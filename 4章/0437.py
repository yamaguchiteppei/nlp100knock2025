import gzip
import json
import re
import math
from collections import Counter, defaultdict
import spacy

# ======== Wikipedia マークアップ除去 ========
def clean_wiki(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r'\[\[([^|\]]+)\|([^]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^]]+)\]\]', r'\1', text)
    text = re.sub(r'\[https?://[^\s]+ (\S+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{\{lang\|[^|]+\|([^}]+)\}\}', r'\1', text)
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'==.*?==', '', text)
    return text

# ======== GiNZA ロード ========
nlp = spacy.load("ja_ginza")

# ======== JSON.gz のパス ========
path = r"C:\Users\哲平\workspace\nlp100knock2025\4章\jawiki-country.json.gz"

# ======== 全体のドキュメント数 ========
documents = []
japan_text = None

# ======== 全記事を走査 ========
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)

        title = article["title"]
        text = clean_wiki(article["text"])

        documents.append(text)

        if title == "Japan" or title == "日本":
            japan_text = text

# ======== 形態素解析して TF 計算（日本の記事のみ） ========
tf = Counter()
doc = nlp(japan_text)

for token in doc:
    if token.pos_ == "NOUN":
        tf[token.lemma_] += 1

# ======== IDF 計算 ========
df = defaultdict(int)
N = len(documents)

for text in documents:
    doc = nlp(text)
    seen = set()

    for token in doc:
        if token.pos_ == "NOUN":
            seen.add(token.lemma_)

    for w in seen:
        df[w] += 1

idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in df}

# ======== TF-IDF 計算 ========
tfidf = {w: tf[w] * idf.get(w, 0) for w in tf}

# ======== 上位20語を表示 ========
top20 = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:20]

print("===== TF-IDF 上位20語 =====")
for word, score in top20:
    print(f"{word:20}  TF={tf[word]:3}  IDF={idf[word]:.3f}  TF-IDF={score:.3f}")
