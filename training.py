import os
import yaml
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils import prepare_data, timeSince, train_epoch, get_dataloader, get_device
from models import Encoder, Decoder, EncoderAttention, DecoderAttention

# Load the dataset
# convert tokens to tensors

# Load YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def train(train_dataloader, encoder, decoder, n_epochs, save_dir, print_every):
    start = time.time()
    print_total_loss = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config['learning_rate'])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config['learning_rate'])
    
    criterion = nn.CrossEntropyLoss(ignore_index=config['special_tokens']['PAD'])

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(1, n_epochs + 1), desc="Training", unit="epoch")

    for epoch in progress_bar:
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, self_attention=True)

        print_total_loss += loss

        # Update the progress bar with current loss and accuracy
        progress_bar.set_postfix({
            'Loss': f'{loss:.4f}',  # Display current loss
        })

        if epoch % print_every == 0:
            print_loss_avg = print_total_loss / print_every
            print_total_loss = 0
            
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

    # Save the trained models
    encoder_path = os.path.join(save_dir, "encoder.pth")
    decoder_path = os.path.join(save_dir, "decoder_masked.pth")

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    print(f"Models saved to {encoder_path} and {decoder_path}")


if __name__ == "__main__":

    batch_size = config['batch_size']
    dataset_path = config['dataset_path']

    title_vocab, content_vocab, train_datloader = get_dataloader(batch_size, dataset_path)

    input_vocab_size = content_vocab.n_words
    output_vocab_size = title_vocab.n_words
    max_sentence_len_target = title_vocab.max_sent_len
    max_sentence_len_input = content_vocab.max_sent_len

    device = get_device()

    encoder = EncoderAttention(hidden_size=config['hidden_size'], input_size=input_vocab_size, max_sen_len=max_sentence_len_input+1).to(device)
    decoder = DecoderAttention(hidden_size=config['hidden_size'], output_size=output_vocab_size, max_sen_len=max_sentence_len_target+1).to(device)

    train(train_dataloader=train_datloader, encoder=encoder, decoder=decoder, n_epochs=config['n_epochs'], save_dir=config['save_dir'], print_every=config['print_every'])