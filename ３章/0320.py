import gzip
import json

uk_text = None

with gzip.open("jawiki-country.json.gz","rt",encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)
        if article["title"] == "イギリス":
             uk_text = article["text"]
             break

if uk_text:
    with open("uk_text.txt","w") as out:
            out.write(uk_text)