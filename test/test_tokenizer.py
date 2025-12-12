
from tokenizers import Tokenizer

path = "../clip-vit-large-patch14-tokenizer.json"
print("Loading:", path)
tok = Tokenizer.from_file(path)
ids = tok.encode("hello world").ids
print("ok, sample ids:", ids[:10])
