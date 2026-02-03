from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

result = fill_mask("The movie was full of [MASK].")

print(f"{result[0]['token_str']}")
