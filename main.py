from preprocessor import create_dataloader

with open("the-verdict.txt", "r") as file:

    raw_text = file.read()

dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)