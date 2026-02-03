from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased",top_k=10)

result = fill_mask("The movie was full of [MASK].")

for r in result:
    print(f"{r['token_str']}\t{r['score']:.3f}")
