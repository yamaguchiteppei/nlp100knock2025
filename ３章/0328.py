import re

info = {}

with open("uk_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- 基礎情報テンプレートの抽出 ---
template = re.search(r'\{\{基礎情報.*?\n(.*?)\n\}\}', text, re.DOTALL)
if template:
    body = template.group(1)

    # --- (1) 強調マークアップ除去 ---
    body = re.sub(r"'{2,5}", "", body)

    # --- (2) 内部リンクの除去 ---
    # [[A|B]] → B
    body = re.sub(r'\[\[([^|\]]+)\|([^]]+)\]\]', r'\2', body)
    # [[A]] → A
    body = re.sub(r'\[\[([^]]+)\]\]', r'\1', body)

    # --- (3) 外部リンクの除去 ---
    # [http://... 説明] → 説明
    body = re.sub(r'\[https?://[^\s]+ (\S+)\]', r'\1', body)
    # [http://...] → (削除)
    body = re.sub(r'\[https?://[^\]]+\]', '', body)

    # --- (4) <ref>タグの除去 ---
    body = re.sub(r'<ref[^>]*>.*?</ref>', '', body)
    body = re.sub(r'<ref[^>]*/>', '', body)

    # --- (5) その他HTMLタグを除去 ---
    body = re.sub(r'<[^>]+>', '', body)

    # --- (6) {{lang|xx|YYY}} の除去（langテンプレート） ---
    body = re.sub(r'\{\{lang\|[^|]+\|([^}]+)\}\}', r'\1', body)

    # --- (7) |key = value の抽出 ---
    fields = re.findall(r'^\|([^=]+?)\s*=\s*(.*)', body, re.MULTILINE)
    for key, value in fields:
        info[key.strip()] = value.strip()

# 結果出力
for k, v in info.items():
    print(k, ":", v)

"""実行結果
略名 : イギリス
日本語国名 : グレートブリテン及び北アイルランド連合王国
公式国名 : United Kingdom of Great Britain and Northern Ireland英語以外での正式国名:
国旗画像 : Flag of the United Kingdom.svg
国章画像 : 85px|イギリスの国章
国章リンク : （国章）
標語 : Dieu et mon droit（フランス語:神と我が権利）
国歌 : God Save the Queen{{en icon}}神よ女王を護り賜え{{center|ファイル:United States Navy Band - God Save the Queen.ogg}}
地図画像 : Europe-UK.svg
位置画像 : United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
公用語 : 英語
首都 : ロンドン（事実上）
最大都市 : ロンドン
元首等肩書 : 女王
元首等氏名 : エリザベス2世
首相等肩書 : 首相
首相等氏名 : ボリス・ジョンソン
他元首等肩書1 : 貴族院議長
他元首等氏名1 : ノーマン・ファウラー
他元首等肩書2 : 庶民院議長
他元首等氏名2 : {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
他元首等肩書3 : 最高裁判所長官
他元首等氏名3 : ブレンダ・ヘイル
面積順位 : 76
面積大きさ : 1 E11
面積値 : 244,820
水面積率 : 1.3%
人口統計年 : 2018
人口順位 : 22
人口大きさ : 1 E7
人口値 : 6643万5600
人口密度値 : 271
GDP統計年元 : 2012
GDP値元 : 1兆5478億
GDP統計年MER : 2012
GDP順位MER : 6
GDP値MER : 2兆4337億
GDP統計年 : 2012
GDP順位 : 6
GDP値 : 2兆3162億
GDP/人 : 36,727
建国形態 : 建国
確立形態1 : イングランド王国／スコットランド王国（両国とも1707年合同法まで）
確立年月日1 : 927年／843年
確立形態2 : グレートブリテン王国成立（1707年合同法）
確立年月日2 : 1707年{{0}}5月{{0}}1日
確立形態3 : グレートブリテン及びアイルランド連合王国成立（1800年合同法）
確立年月日3 : 1801年{{0}}1月{{0}}1日
確立形態4 : 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
確立年月日4 : 1927年{{0}}4月12日
通貨 : UKポンド (£)
通貨コード : GBP
時間帯 : ±0
夏時間 : +1
ISO 3166-1 : GB / GBR
ccTLD : .uk / .gb
国際電話番号 : 44
注記 :
"""