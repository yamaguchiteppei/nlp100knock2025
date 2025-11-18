import re
import requests

# --- User-Agent 必須 ---
headers = {
    "User-Agent": "NLP100/1.0 (contact: yamaguchi@example.com)"
}

with open("uk_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

template = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, re.DOTALL).group(1)

match = re.search(r"\|国旗画像\s*=\s*(.+)", template)
filename = match.group(1).strip()

URL = "https://ja.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "titles": f"File:{filename}",
    "prop": "imageinfo",
    "iiprop": "url",
    "format": "json"
}

res = requests.get(URL, params=params, headers=headers)
print("status:", res.status_code)
data = res.json()

pages = data["query"]["pages"]
for page_id, page in pages.items():
    print(page["imageinfo"][0]["url"])

"""実行結果
status: 200
https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg
"""