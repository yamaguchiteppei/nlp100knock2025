import re
import spacy
from collections import Counter
import pandas as pd
from tqdm import tqdm

open_file = "4章/jawiki-country.json.gz"

# === GiNZAロード（重いNERとParserを無効化） ===
nlp = spacy.load("ja_ginza")
nlp.disable_pipes("ner", "parser")

# === 正規表現の事前コンパイル ===
patterns = [
    r"'{2,5}",                     # 強調
    r"\[\[(?:[^|\]]*\|)?([^|\]]+)\]\]",  # 内部リンク
    r"\[(https?://\S+)(?: \S+)?\]",       # 外部リンク
    r"<[^>]+>",                    # HTMLタグ
    r"\{\{.*?\}\}",                # テンプレート
]
compiled_pattern = re.compile("|".join(patterns), re.MULTILINE)

def remove_markup(text):
    """Wikipediaマークアップを再帰的に除去"""
    old = ""
    while old != text:
        old = text
        text = compiled_pattern.sub(lambda m: m.group(1) if m.lastindex else "", text)
    return text

def split_text_by_bytes(text, max_bytes=49149):
    """Sudachi制限回避のためUTF-8バイト長で分割"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    chunks = []
    for i in range(0, len(encoded), max_bytes):
        chunk = encoded[i:i + max_bytes]
        chunks.append(chunk.decode("utf-8", errors="ignore"))
    return chunks

def main():
    # ===== Wikipedia JSON 読み込み =====
    df = pd.read_json(open_file, lines=True, compression="gzip")
    
    # ===== マークアップ除去 =====
    df["cleaned"] = df["text"].apply(remove_markup)

    # ===== Sudachi対策：記事を分割 =====
    df["chunks"] = df["cleaned"].apply(split_text_by_bytes)

    # 行へ展開
    text_series = df.explode("chunks")["chunks"]
    text_series = text_series[text_series.str.strip() != ""]

    # ===== 品詞のターゲット =====
    target_pos = {"NOUN", "VERB", "ADJ", "ADV"}

    all_words = []

    tqdm_series = tqdm(text_series, desc="形態素解析中", total=len(text_series))

    # ===== GiNZAパイプで高速解析 =====
    docs = nlp.pipe(
        tqdm_series,
        batch_size=200,
        n_process=4,
    )

    for doc in docs:
        for token in doc:
            if token.pos_ in target_pos:
                all_words.append(token.lemma_)   # ← 辞書形でカウント

    # ===== 出現頻度 =====
    counts = Counter(all_words).most_common(20)

    print("単語\t頻度")
    for w, c in counts:
        print(f"{w}\t{c}")

if __name__ == "__main__":
    main()

"""実行結果
単語	出現頻度
年	27900
いる	13325
月	11892
ある	11615
日	7762


	7534
人	6463
なっ	4680
こと	4552
%	4215
い	3769
世界	3569
あり	3479
語	3262
|	3051
ため	3046
政府	3031
島	3023
第	3017
大統領	2938
"""