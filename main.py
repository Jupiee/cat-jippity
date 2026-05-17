from preprocessor import create_dataloader
from core import GPTModel, generate_text_simple
import torch, tiktoken

vocab_size = 50257
output_dim = 256
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

with open("the-verdict.txt", "r") as file:

    raw_text = file.read()

max_length = 4

dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)

data_iter = iter(dataloader)
first_batch_input, first_batch_output = next(data_iter)
print(f"Inputs: {first_batch_input}\nOutput: {first_batch_output}")

token_embeddings = embedding_layer(first_batch_input)
print(token_embeddings.shape)

context_length = max_length
positional_emb_layer = torch.nn.Embedding(context_length, output_dim)
positional_embeddings = positional_emb_layer(torch.arange(context_length))
print(positional_embeddings.shape)

input_embeddings = token_embeddings + positional_embeddings
print(input_embeddings.shape)

if __name__ == '__main__':

    def text_to_tokens(text, tokenizer):

        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        return encoded_tensor

    def token_to_text(token_ids, tokenizer):

        flat = token_ids.squeeze(0)
        
        return tokenizer.decode(flat.tolist())

    GPT_CONFIG_124M = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 256, # Context length
        "embedding_dim": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of layers
        "drop_rate": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias
    }

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    tokens = text_to_tokens(start_context, tokenizer)

    token_ids = generate_text_simple(
        model,
        tokens,
        10,
        GPT_CONFIG_124M["context_length"]
    )

    print(f"Output text:\n{token_to_text(token_ids, tokenizer)}")