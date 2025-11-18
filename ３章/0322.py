import re
with open("uk_text.txt","r",encoding="utf-8") as f:
    for line in f:
        match = re.match(r"\[\[Category:(.+?)\]\]",line)
        if match:
            print(match.group(1))

"""実行結果
イギリス|*
イギリス連邦加盟国
英連邦王国|*
G8加盟国
欧州連合加盟国|元
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
"""