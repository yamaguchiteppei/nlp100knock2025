import spacy 

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

nlp = spacy.load("ja_ginza")
doc = nlp(text)

for sent in doc.sents:
    for token in sent:
        print(f"{token.text}\t{token.head}\t{token.dep_}")

"""実行結果
        メロス  compound
メロス  激怒    nsubj
は      メロス  case
激怒    激怒    ROOT
し      激怒    aux
た      激怒    aux
。      激怒    punct


        ROOT
必ず    除か    advmod
、      必ず    punct
かの    暴虐    det
邪智    暴虐    compound
暴虐    王      nmod
の      暴虐    case
王      除か    obj
を      王      case
除か    決意    advcl
なけれ  除か    aux
ば      なけれ  fixed
なら    なけれ  fixed
ぬ      なけれ  fixed
と      除か    case
決意    決意    ROOT
し      決意    aux
た      決意    aux
。      決意    punct

        メロス  compound
メロス  わから  obl
に      メロス  case
は      メロス  case
政治    わから  nsubj
が      政治    case
わから  わから  ROOT
ぬ      わから  aux
。      わから  punct

        メロス  compound
メロス  牧人    nsubj
は      メロス  case
、      メロス  punct
村      牧人    nmod
の      村      case
牧人    牧人    ROOT
で      牧人    cop
ある    で      fixed
。      牧人    punct

        笛      compound
笛      吹き    obj
を      笛      case
吹き    暮し    advcl
、      吹き    punct
羊      遊ん    obl
と      羊      case
遊ん    暮し    advcl
で      遊ん    mark
暮し    暮し    ROOT
て      暮し    mark
来      て      fixed
た      暮し    aux
。      暮し    punct

        邪悪    obl
けれど
        case
も
        case
邪悪    敏感    obl
に      邪悪    case
対し    に      fixed
ては    に      fixed
、      邪悪    punct
人      倍      compound
一      倍      nummod
倍      敏感    obl
に      倍      case
敏感    敏感    ROOT
で      敏感    aux
あっ    で      fixed
た      敏感    aux
。      敏感    punct


        ROOT
"""