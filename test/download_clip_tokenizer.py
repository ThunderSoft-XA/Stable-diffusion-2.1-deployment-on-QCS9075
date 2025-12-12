from tokenizers import Tokenizer

if __name__ == "__main__":
    tok = Tokenizer.from_pretrained("openai/clip-vit-large-patch14")
    out_path = "clip-vit-large-patch14-tokenizer.json"
    tok.save(out_path)
    print(f"Saved tokenizer to {out_path}")
