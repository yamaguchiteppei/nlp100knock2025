import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

prompt = """9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。"""
response = model.generate_content(prompt)

print(response.text)

"""実行結果
年代の古い順に並べると以下のようになります。

*   **イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。**
    *   これは**810年**（薬子の変の直後）のできごとです。

*   **ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。**
    *   承和の変は**842年**のできごとです。

*   **ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。**
    *   これは**901年**（昌泰の変）のできごとです。厳密には10世紀初頭ですが、菅原道真や藤原時平が9世紀末に活躍していた人物であり、9世紀の政治の延長線上の出来事として捉えられます。

したがって、年代の古い順に並べると、**イ → ウ → ア** となります。
"""