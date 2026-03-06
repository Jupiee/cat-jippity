from preprocessor import create_dataloader
import torch

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