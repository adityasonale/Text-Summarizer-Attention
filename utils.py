import re
import os
import time
import yaml
import math
import unicodedata

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Load YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

SOS_token = config['special_tokens']['SOS']
EOS_token = config['special_tokens']['EOS']
PAD_token = config['special_tokens']['PAD']
UNK_token = config['special_tokens']['UNK']

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def _load_dataset(dataset_path:str):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Warning: Unexpected error loading {dataset_path}. Error: {e}")
        return None
    
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def _preprocess_sentence(sentence:str):

    # Convert this to lowercase
    sentence = sentence.lower()

    # Convert unicode to ascii
    sentence = unicodeToAscii(sentence)

    # Remove URLs
    sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)

    # Normalize sentence
    sentence = _normalize_prefix(sentence)

    # Remove punctuation except sentence boundaries
    sentence = re.sub(r'[^\w\s.]', '', sentence)
    
    # Remove extra whitespace
    sentence = ' '.join(sentence.split())

    return sentence

def _normalize_prefix(sentence):
    replacements = {
        "i m ": "i am ",
        "he s ": "he is ",
        "she s ": "she is ",
        "you re ": "you are ",
        "we re ": "we are ",
        "they re ": "they are "
    }

    for k,v in replacements.items():
        sentence = re.sub(r'^' + re.escape(k), v, sentence)
    return sentence

class SummarizerVocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
        self.n_words = 4
        self.freq_word = {}
        self.max_sent_len = 0

    def addSentence(self, sentence):
        sentence = _preprocess_sentence(sentence)
        sentence = sentence.split(' ')
        if len(sentence) > self.max_sent_len:
            self.max_sent_len = len(sentence)

        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.freq_word[word] = 1
            self.n_words += 1

        else:
            self.freq_word[word] += 1

class SummarizationDataset(Dataset):
    def __init__(self, title_tensors, content_tensors):
        self.title_tensors = title_tensors
        self.content_tensors = content_tensors

    def __len__(self):
        return len(self.title_tensors)

    def __getitem__(self, idx):
        return self.title_tensors[idx], self.content_tensors[idx]

# def readDataset(dataset):

#     title = dataset['title']
#     content = dataset['content']

#     title_summarize_vocab = SummarizerVocab() # target
#     content_summarize_vocab = SummarizerVocab() # input

#     return title_summarize_vocab, content_summarize_vocab

def prepare_data(dataset_path:str):

    dataset = _load_dataset(dataset_path)

    title = dataset['title']
    content = dataset['content']

    title_vocab = SummarizerVocab()
    content_vocab = SummarizerVocab()

    for i, j in zip(title, content):
        title_vocab.addSentence(i)
        content_vocab.addSentence(j)

    print("Printing tokens for title and content")
    print("Total tokens in Title: ", title_vocab.n_words)
    print("Total tokens in Content: ", content_vocab.n_words)

    # Convert to tensors
    title_tensors = [sent_to_tensor(_preprocess_sentence(s), title_vocab) for s in title]
    content_tensors = [sent_to_tensor(_preprocess_sentence(s), content_vocab) for s in content]

    title_tensors_padded = pad_sequence(title_tensors, batch_first=True, padding_value=PAD_token)
    content_tensors_padded = pad_sequence(content_tensors, batch_first=True, padding_value=PAD_token)

    return title_vocab, content_vocab, title_tensors_padded, content_tensors_padded


def sent_to_tensor(sentence:str, summarize_vocab:SummarizerVocab):
    sentence = sentence.split()
    indices = []

    indices = [
        summarize_vocab.word2idx.get(word)
        for word in sentence
    ]

    indices.append(EOS_token)

    return torch.tensor(indices, dtype=torch.long)
    
def get_dataloader(batch_size, dataset_path, shuffle=True):
    title_vocab, content_vocab, title_tensors, content_tensors = prepare_data(dataset_path)

    dataset = SummarizationDataset(title_tensors, content_tensors)

    return title_vocab, content_vocab, DataLoader(dataset, batch_size, shuffle=shuffle)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, self_attention:bool=False):

    total_loss = 0
    device = get_device()
    for data in dataloader:

        title_tensor, content_tensor = data

        title_tensor = title_tensor.to(device)
        content_tensor = content_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        if self_attention:
            encoder_outputs = encoder(content_tensor)
            decoder_input = title_tensor[:, :-1]  # Remove last token
            decoder_target = title_tensor[:, 1:]  # Remove first token (usually SOS)
            decoder_outputs = decoder(encoder_outputs, decoder_input)
            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), 
                           decoder_target.contiguous().view(-1))
        else:
            encoder_outputs, encoder_hidden = encoder(content_tensor)
            decoder_outputs, decoder_hidden = decoder(encoder_outputs, encoder_hidden)

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), title_tensor.view(-1))

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    
    prepare_data(r"D:\Datasets\news_data\text_summarizer_data.csv")