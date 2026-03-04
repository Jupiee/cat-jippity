import re, torch, tiktoken
from torch.utils.data import Dataset, DataLoader

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
        
class GPTDataset(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Note to self:
    # stride is the offset of second batch's input
    # text: In the heart of the city stood the old library, a relic from a bygone era. 
    # batch 1 input: In the heart
    # batch 2 input: the heart of the

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
