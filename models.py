import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device

import yaml
import matplotlib.pyplot as plt

# Load YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

SOS_token = config['special_tokens']['SOS']
EOS_token = config['special_tokens']['EOS']
PAD_token = config['special_tokens']['PAD']

# Getting device
device = get_device()

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        # Defining Key, Query, Value
        self.Key_W = nn.Linear(hidden_size, hidden_size)
        self.Query_W = nn.Linear(hidden_size, hidden_size)
        self.Value_W = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, hidden_states):
        
        # Dot product between Query and Key

        # query input shape: (batch_size, query_sequence_len, hidden_size)
        # hidden_state shape: (batch_size, key_sequence_len, hidden_size)

        key = self.Key_W(hidden_states)
        query = self.Query_W(query)
        value = self.Value_W(hidden_states)

        key_reshaped = key.permute(0, 2, 1)

        attn_output = torch.bmm(query, key_reshaped) / self.hidden_size**0.5 # (batch_size, sequence_len, sequence_len)

        attn_score = F.softmax(attn_output, dim=-1) # (batch_size, sequence_len, sequence_len)

        output = torch.bmm(attn_score, value)

        return output # (batch_size, sequence_len, hidden_size)
    
class AttentionMasked(nn.Module):
    def __init__(self, hidden_size, masked:bool):
        super(AttentionMasked, self).__init__()

        # Define weights
        self.query_W = nn.Linear(hidden_size, hidden_size)
        self.key_W = nn.Linear(hidden_size, hidden_size)
        self.value_W =nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.masked = masked
        self.device = get_device()

    def forward(self, query, encoder_hidden_states=None):

        if encoder_hidden_states is None:
            encoder_hidden_states = query

        query = self.query_W(query)
        key = self.key_W(encoder_hidden_states)
        value = self.value_W(encoder_hidden_states)

        # Calculate attention scores
        
        # query_shape = (batch_size, query_seq_len, hidden_size)
        # key_shape = (batch_size, seq_len, hidden_size)

        key_transposed = key.permute(0, 2, 1)

        attention = torch.bmm(query, key_transposed) / (self.hidden_size**0.5)

        if self.masked:
            seq_len = query.size(1)
            mask = self.create_look_ahead_mask(seq_len).to(self.device)
            attention = attention.masked_fill(mask==0, float('-inf'))

        attention_score = F.softmax(attention, dim=-1)

        attention_output = torch.bmm(attention_score, value)

        return attention_output, attention_score
    
    def create_look_ahead_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        return mask
        
class Encoder(nn.Module):
    def __init__(self, hidden_size, input_size, n_layers, dropout_p=0.5):
        super(Encoder, self).__init__()

        self.n_layers = n_layers

        # Defining layers

        # 1. Embedding Layer (To make embeddings for each token in a sequence)
        # 2. LSTM Layer (To store sequential and positional information)
        # 3. Dropout Layer (Regularization and reduce overfitting)

        self.embdding = nn.Embedding(input_size, hidden_size)
        self.enc_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        
        # input shape: (batch_size, seq_len)
        x = self.dropout(self.embdding(x)) # output: (batch_size, seq_len, embedding_dim)
        
        # input_shape: (batch_size, seq_len, embedding_dim)
        x, hidden = self.enc_lstm(x) # output: (batch_size, seq_len, hidden_size)

        return x, hidden
    
class EncoderAttention(nn.Module):
    def __init__(self, hidden_size, input_size, max_sen_len, dropout_p=0.5):
        super().__init__()

        self.token_embedding = nn.Embedding(input_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_sen_len, hidden_size)
        self.self_attention = AttentionMasked(hidden_size, masked=False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        batch_size, seq_len = x.size()

        position = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.positional_embedding(position)
        x = self.dropout(x)
        x, _ = self.self_attention(x)

        return x # (batch_size, seq_len, embedding_dim)

class Decoder(nn.Module):

    def __init__(self, output_size, hidden_size, max_sentence_len, dropout_p=0.5):
        super(Decoder, self).__init__()

        # Defining Layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size*2, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.attention = AttentionMasked(hidden_size, True)
        self.out = nn.Linear(hidden_size, output_size)
        self.max_sent_len = max_sentence_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        
        # Getting the batch size from encoder_outputs
        batch_size = encoder_outputs.size(0)

        # Creating input for decoder
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_sent_len):
            decoder_output, decoder_hidden = self._forward_step(decoder_input, decoder_hidden, encoder_outputs)

            # store the output
            decoder_outputs.append(decoder_output)

            if target_tensor:
                # Using teacher forcing technique.
                decoder_input = target_tensor[:, i].unsqueeze(0)

            else:
                _, topi = decoder_output.topk(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden

    def _forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input)) # output shape: (batch_size, 1, hidden_size)

        # Concatenate attention output with the context vector.
        hidden_state, cell_state = hidden # (num_layers*num_directions, batch_size, hidden_size)

        query = hidden_state.permute(1, 0, 2)

        # encoder_outputs shape: (batch_size, sequence_len, embedding_dim)
        attn_output, attn_weights = self.attention(query, encoder_outputs) # (batch_size, seq_len, hidden_size)

        # concatenate attention output with context vector and pass as input.
        context_vector = torch.cat((embedded, attn_output), dim=2)

        output, hidden = self.decoder_lstm(context_vector, hidden) # (batch_size, seq_len_hidden_size), (num_layers*num_directions, batch_size, hidden_size)

        # Send output to linear layer
        output = self.out(output) # (batch_size, 1, vocab_size)

        return output, hidden
    
class DecoderAttention(nn.Module):
    def __init__(self, output_size, hidden_size, max_sen_len, dropout_p=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_token)
        self.positional_embedding = nn.Embedding(max_sen_len, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.self_attention = AttentionMasked(hidden_size, masked=True)
        self.cross_attention = AttentionMasked(hidden_size, masked=False)

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.max_sent_len = max_sen_len

    def forward(self, encoder_outputs, target_tokens):
        batch_size, seq_len = target_tokens.size()

        # 1. Embedding + Positional Encoding
        token_emb = self.token_embedding(target_tokens)  # (batch_size, seq_len, hidden_size)

        position_ids = torch.arange(seq_len, device=target_tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.positional_embedding(position_ids)  # (batch_size, seq_len, hidden_size)

        x = token_emb + pos_emb

        x = self.dropout(x)

        # 2. Masked Self-Attention (decoder attends to previous positions only)
        x, _ = self.self_attention(x)

        # 3. Cross-Attention (decoder attends to encoder outputs)
        x, _ = self.cross_attention(x, encoder_outputs)

        # 4. Project to vocabulary
        x = self.output_layer(x)  # (batch_size, seq_len, vocab_size)

        return x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    hidden_size = 8

    device = get_device()

    # Random tensors simulating embedded sequences
    query = torch.randn(batch_size, seq_len, hidden_size).to(device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device)

    # Instantiate and run the Attention module
    attention = AttentionMasked(hidden_size , masked=True).to(device)
    output, attn_weights = attention(query, hidden_states)

    # Output shape check
    # print("Output shape:", output.shape)  # Expected: (2, 5, 8)
    print("Attention weights shape:", attn_weights.shape)


    # Plot heatmap for the first sample in the batch
    plt.figure(figsize=(6, 5))
    plt.title("Masked Attention Weights Heatmap")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")

    # Detach, move to CPU, and squeeze batch dimension
    plt.imshow(attn_weights[0].detach().cpu(), cmap='viridis')

    plt.colorbar()
    plt.show()