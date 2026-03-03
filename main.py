from preprocessor import Tokenizer

with open("the-verdict.txt", "r") as file:

    raw_text = file.read()

tokenizer = Tokenizer()

tokenizer.tokenize(raw_text)

text1 = """"The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication."""
text2 = "Hello, do you like tea?"

text = " <|endoftext|> ".join((text1, text2))

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))