from transformers import AlbertModel, AlbertTokenizer
import torch

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple



print(input_ids.size())  # torch.Size([1, 8])
print('------------------------------')
print(input_ids)         # tensor([[    2, 10975,    15,    51,  1952,    25, 10901,     3]])
print('------------------------------')
print(outputs)
print('------------------------------')
print(last_hidden_states)






PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-spiece.model",
        "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v1-spiece.model",
        "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v1-spiece.model",
        "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v1-spiece.model",
        "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model",
        "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-spiece.model",
        "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-spiece.model",
        "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-spiece.model",
    }
}
