import gzip
import json
import re
from collections import Counter
import spacy

# ===== (1) Wikipedia マークアップ除去関数 =====
def clean_wiki(text):
    text = re.sub(r"'{2,5}", "", text)                                 # 強調
    text = re.sub(r'\[\[([^|\]]+)\|([^]]+)\]\]', r'\2', text)          # [[A|B]]
    text = re.sub(r'\[\[([^]]+)\]\]', r'\1', text)                     # [[A]]
    text = re.sub(r'\[https?://[^\s]+ (\S+)\]', r'\1', text)           # 外部リンク
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)                    # <ref>
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<[^>]+>', '', text)                                # HTMLタグ
    text = re.sub(r'\{\{lang\|[^|]+\|([^}]+)\}\}', r'\1', text)        # langテンプレ
    text = re.sub(r'\{\{[^}]+\}\}', '', text)                          # その他テンプレ
    text = re.sub(r'==.*?==', '', text)                                # 見出し
    return text


# ===== (2) GiNZAをロード =====
nlp = spacy.load("ja_ginza")
counter = Counter()

# ===== (3) Wikipedia JSON.gz ファイル =====
path = r"C:\Users\哲平\workspace\nlp100knock2025\4章\jawiki-country.json.gz"

# ===== (4) 巨大ファイル → 記事ごとに安全に解析 =====
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)
        cleaned = clean_wiki(article["text"])

        # 長すぎる文章でGiNZAが落ちないように分割
        parts = re.split(r"\n{2,}", cleaned)

        for part in parts:
            if not part.strip():
                continue

            # GiNZAで解析
            doc = nlp(part)

            # ===== (5) 全品詞の“単語（形態素）”をカウント =====
            for token in doc:
                if token.pos_ not in ["SPACE"]:
                    # surface ではなく lemma_（辞書形）で統一
                    counter[token.lemma_] += 1


# ===== (6) 上位20語を表示 =====
print("=== 出現頻度トップ20 ===")
for word, freq in counter.most_common(20):
    print(word, freq)
