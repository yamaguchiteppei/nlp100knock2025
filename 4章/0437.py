import gzip
import json
import re
import math
from collections import Counter, defaultdict
import spacy

#  Wikipedia マークアップ除去
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

# GiNZA ロード
nlp = spacy.load("ja_ginza")

# JSON.gz のパス 
path = r"C:\Users\哲平\workspace\nlp100knock2025\4章\jawiki-country.json.gz"

#  全体のドキュメント数 
documents = []
japan_text = None

# 全記事を走査 
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)

        title = article["title"]
        text = clean_wiki(article["text"])

        documents.append(text)

        if title == "Japan" or title == "日本":
            japan_text = text

# 形態素解析して TF 計算（日本の記事のみ）
tf = Counter()
doc = nlp(japan_text)

for token in doc:
    if token.pos_ == "NOUN":
        tf[token.lemma_] += 1

# IDF 計算 
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

# TF-IDF 計算 
tfidf = {w: tf[w] * idf.get(w, 0) for w in tf}

# 上位20語を表示 
top20 = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:20]

print("===== TF-IDF 上位20語 =====")
for word, score in top20:
    print(f"{word:20}  TF={tf[word]:3}  IDF={idf[word]:.3f}  TF-IDF={score:.3f}")

"""実行中
単語              TF         IDF        TF-IDF    
年               687        1.0081     692.5404  
天皇              69         3.6842     254.2125  
国               202        1.0925     220.6856  
倭国              33         5.4188     178.8217  
列島              43         3.8784     166.7710  
月               162        1.0203     165.2861  
こと              155        1.0368     160.7062  
第               144        1.1014     158.5948  
世界              136        1.0707     145.6173  
関係              121        1.1562     139.8954  
倭               25         5.4188     135.4710  
大日本             33         3.8784     127.9871  
頁               57         2.1867     124.6430  
日本書紀            21         5.8243     122.3104  
人               108        1.0368     111.9759  
位               76         1.4612     111.0517  
問題              77         1.3995     107.7583  
経済              90         1.1148     100.3298  
国号              31         3.2216     99.8701   
憲法              75         1.3301     99.7550
"""