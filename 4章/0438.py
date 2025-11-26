import re
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

open_file = "jawiki-country.json.gz"
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

    # チャンク分割 (長さ制限対策)
    df['chunks'] = df['cleaned_text'].apply(split_text_by_bytes)
    # 記事IDを保持するためにindexをリセットして列にする（後で結合するため）
    df = df.reset_index().rename(columns={'index': 'article_id'})
    
    # チャンクを行に展開
    exploded_df = df.explode('chunks').reset_index(drop=True)
    
    # 空のチャンクを除外
    text_series = exploded_df['chunks'].fillna('')
    text_series = text_series[text_series.str.len() > 0]
    
    target_pos = {'NOUN'}
    noun_lists = []
    
    # nlp.pipeで一括処理
    docs = nlp.pipe(tqdm(text_series, desc="形態素解析"), batch_size=200, n_process=4, disable=['ner', 'parser'])
    
    for doc in docs:
        nouns = [token.text for token in doc if token.pos_ in target_pos]
        noun_lists.append(" ".join(nouns))
    
    # 解析結果をexploded_dfに戻す
    exploded_df.loc[text_series.index, 'nouns'] = noun_lists
    
    # 記事単位に再結合
    exploded_df['nouns'] = exploded_df['nouns'].fillna('')
    # 記事ごとに名詞をスペース区切りで結合
    article_nouns = exploded_df.groupby('article_id')['nouns'].apply(lambda x: " ".join(x))
    
    # 元のdfに結合
    df['nouns'] = article_nouns
    
    # TF-IDF計算
    print("TF-IDF計算中...")
    # CountVectorizerでTF（単語出現回数）を取得
    # token_patternで1文字の単語も許容する
    count_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    tf_matrix = count_vectorizer.fit_transform(df['nouns'])
    feature_names = np.array(count_vectorizer.get_feature_names_out())

    # TfidfTransformerでIDFとTF-IDFを計算
    # norm=None: 正規化しない（単純な TF * IDF にする）
    tfidf_transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)
    
    # 「日本」の記事を特定して表示
    target_title = '日本'
    target_df = df[df['title'] == target_title]
    
    if target_df.empty:
        print(f"記事「{target_title}」が見つかりませんでした。")
        return

    target_index = target_df.index[0]
    
    # 対象記事のベクトルを取得
    tf_vector = tf_matrix[target_index]
    tfidf_vector = tfidf_matrix[target_index]
    
    # 疎行列からデータを取り出す
    coo = tfidf_vector.tocoo()
    
    results = []
    for idx, score in zip(coo.col, coo.data):
        word = feature_names[idx]
        tf = tf_vector[0, idx] # 生の出現回数
        idf = tfidf_transformer.idf_[idx]
        results.append((word, tf, idf, score))
    
    # TF-IDFスコアで降順ソート
    results.sort(key=lambda x: x[3], reverse=True)
    
    # 上位20件を表示
    print(f"記事「{target_title}」のTF-IDF上位20語")
    print(f"{'単語':<15} {'TF':<10} {'IDF':<10} {'TF-IDF':<10}")
    for word, tf, idf, score in results[:20]:
        print(f"{word:<15} {tf:<10} {idf:<10.4f} {score:<10.4f}")

if __name__ == "__main__":
    main()

"""実行結果
  TF-IDF計算中...
記事「日本」のTF-IDF上位20語
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