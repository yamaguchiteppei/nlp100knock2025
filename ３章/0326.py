import re 
with open("uk_text.txt","r",encoding="utf-8") as f:
    sentence = f.read()
    
template = re.search(r'\{\{基礎情報.*?\n(.*?)\n\}\}',sentence,re.DOTALL)
if template:
    lines = template.group(1)

    fixed_text = re.sub(r"'{5}","",lines)
    fixed_text = re.sub(r"'{3}","",fixed_text)
    fixed_text = re.sub(r"'{2}","",fixed_text)
    print(fixed_text)

"""実行結果
|GDP統計年MER = 2012
|GDP順位MER = 6
|GDP値MER = 2兆4337億<ref name="imf-statistics-gdp" />
|GDP統計年 = 2012
|GDP順位 = 6
|GDP値 = 2兆3162億<ref name="imf-statistics-gdp" />
|GDP/人 = 36,727<ref name="imf-statistics-gdp" />
|建国形態 = 建国
|確立形態1 = [[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）
|確立年月日1 = 927年／843年
|確立形態2 = [[グレートブリテン王国]]成立<br />（1707年合同法）
|確立年月日2 = 1707年{{0}}5月{{0}}1日
|確立形態3 = [[グレートブリテン及びアイルランド連合王国]]成立<br />（[[合同法 (1800年)|1800年合同法]]）
|確立年月日3 = 1801年{{0}}1月{{0}}1日
|確立形態4 = 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
|確立年月日4 = 1927年{{0}}4月12日
|通貨 = [[スターリング・ポンド|UKポンド]] (£)
|通貨コード = GBP
|時間帯 = ±0
|夏時間 = +1
|ISO 3166-1 = GB / GBR
|ccTLD = [[.uk]] / [[.gb]]<ref>使用は.ukに比べ圧倒的少数。</ref>
|国際電話番号 = 44
|注記 = <references/>
"""