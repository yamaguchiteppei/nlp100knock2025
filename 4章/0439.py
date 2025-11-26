import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

open_file = "4章/jawiki-country.json.gz"
nlp = spacy.load('ja_ginza')

# 正規表現パターンを事前にコンパイル
patterns = [
    r"\'{2,5}",  # 強調マークアップ
    r"\[\[(?:[^|\]]*?\|)?([^|\]]+?)\]\]",  # 内部リンク
    r"\[(https?://[^ ]+)( [^\]]+)?\]",  # 外部リンク
    r"<[^>]+>",  # HTMLタグ
    r"\{\{.*?\}\}",  # テンプレート
]
compiled_pattern = re.compile('|'.join(patterns), re.MULTILINE)

def remove_markup(text):
    old_text = ""
    while old_text != text:
        old_text = text
        text = compiled_pattern.sub(lambda m: m.group(1) if m.group(1) else '', text)
    return text

def split_text_by_bytes(text, max_bytes=49149):
    """Sudachiの制限に合わせてテキストを分割しリストで返す"""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return [text]
    
    chunks = []
    for i in range(0, len(encoded), max_bytes):
        chunk = encoded[i:i + max_bytes]
        chunks.append(chunk.decode('utf-8', errors='ignore'))
    return chunks

def main():
    df = pd.read_json(open_file, orient='records', lines=True, compression='gzip')
    
    df['cleaned_text'] = df['text'].apply(remove_markup)
    
    df['chunks'] = df['cleaned_text'].apply(split_text_by_bytes)
    
    # チャンクを行に展開
    text_series = df.explode('chunks')['chunks']
    text_series = text_series[text_series.str.len() > 0]
    
    # 形態素解析 (単語抽出)
    # Zipfの法則は一般的にすべての単語を対象にするが，ここでは意味のある語に絞る
    target_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    all_words = []
    
    docs = nlp.pipe(tqdm(text_series, desc="形態素解析"), batch_size=200, n_process=4, disable=['ner', 'parser'])
    
    for doc in docs:
        for token in doc:
            if token.pos_ in target_pos:
                all_words.append(token.text)
    
    # 出現頻度の集計
    word_counts = Counter(all_words)
    
    # 頻度順にソート (単語, 頻度) のリスト
    sorted_counts = word_counts.most_common()
    
    # 順位と頻度のリストを作成
    ranks = range(1, len(sorted_counts) + 1)
    frequencies = [count for word, count in sorted_counts]
    
    # グラフのプロット
    plt.figure(figsize=(10, 6))
    plt.scatter(ranks, frequencies, s=10, alpha=0.5)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title('Zipf\'s Law (Log-Log Plot)')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 画像として保存
    output_img = 'zipf_law.png'
    plt.savefig(output_img)
    print(f"グラフを {output_img} に保存しました")
    
    # plt.show()

if __name__ == "__main__":
    main()