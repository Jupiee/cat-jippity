import re

class Tokenizer:

    def __init__(self):
        
        self._vocabulary = None
        self._id_to_token = None

    def tokenize(self, raw_text):

        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
        tokenized_text = [token.strip() for token in preprocessed if token.strip()]
        sorted_unique_words = sorted(list(set(tokenized_text)))
        sorted_unique_words.extend(["<|endoftext|>", "<|unk|>"])

        self._vocabulary = {token:idx for idx, token in enumerate(sorted_unique_words)}
        self._id_to_token = {idx:token for token, idx in self._vocabulary.items()}
    
    def encode(self, raw_text: str):

        if self._vocabulary is not None:
            
            split_text = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
            tokenized_text = [token.strip() for token in split_text if token.strip()]
            tokenized_text = [token if token in self._vocabulary else "<|unk|>" for token in tokenized_text]
            
            token_ids = [self._vocabulary[token] for token in tokenized_text]

            return token_ids
    
    def decode(self, ids):

        if self._id_to_token is not None:

            text = " ".join([self._id_to_token[id] for id in ids])
            decoded_texts = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return decoded_texts
        
    def vocabulary_size(self):

        if self._vocabulary is not None:

            return len(self._vocabulary.items())

class Preprocessor:

    ...
