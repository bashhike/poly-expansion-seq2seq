import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import * 

class Seq2seqBaseline(nn.Module):
    def __init__(self, vocab, emb_dim = 6, hidden_dim = 256, num_layers = 3, dropout=0.1):
        super().__init__()        
        self.num_words = num_words = vocab.num_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=emb_dim)
        self.encoder_gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.decoder_gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=False)
        self.decoder_linear = nn.Linear(in_features=hidden_dim, out_features=num_words)

    def encode(self, source):
        """
        Encode the source batch using a bidirectional GRU encoder.
        """

        # Compute a tensor containing the length of each source sequence.
        source_lengths = torch.sum(source != pad_id, axis=0).cpu()

        encoder_mask = ~source.eq(pad_id)

        embeds = self.embedding(source)
        # Pack padded sequences for better throughput 
        packed_embeds = pack_padded_sequence(embeds, source_lengths)
        
        packed_output, hidden = self.encoder_gru(packed_embeds)
        output, input_sizes = pad_packed_sequence(packed_output)
        
        # Need to reduce the first dimension of hidden state by half (num_layers*2, batch_size, hidden_dim)
        encoder_hidden = hidden[:self.num_layers, :, :] + hidden[self.num_layers:, :, :]
        
        return output, encoder_mask, encoder_hidden


    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        """Run the decoder GRU for one decoding step from the last hidden state.
        """

        # These arguments are not used in the baseline model.
        del encoder_output
        del encoder_mask

        decoder_embed = self.embedding(decoder_input)
        decoder_relu = F.relu(decoder_embed) 
        output, hidden = self.decoder_gru(decoder_relu, last_hidden)
        logits = self.decoder_linear(output)
        return logits[0], hidden, None

        

    def compute_loss(self, source, target):
        """Run the model on the source and compute the loss on the target using teacher forcing. 
           """
        seq_length = target.shape[0]
        encoded_sequence, encoder_mask, encoder_hidden = self.encode(source)
        
        # Ignore the pad tokens while calculating the loss 
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss = 0.0
        decoder_hidden = encoder_hidden
        for i in range(seq_length-1):
            step_x = target[[i], :]
            step_target = target[[i+1], :]
            logits, decoder_hidden, attention_weights = self.decode(step_x, decoder_hidden, encoded_sequence, encoder_mask)
            loss += criterion(logits, step_target[0])
        return loss/(seq_length-1)



class Seq2seqAttention(Seq2seqBaseline):
    def __init__(self, vocab):
        super().__init__(vocab)

        # Initialize any additional parameters needed for this model that are not
        # already included in the baseline model.
        
        self.attention = nn.Linear(self.hidden_dim*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.l_reshape = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.l_output = nn.Linear(self.hidden_dim*2, self.num_words)

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        """Run the decoder GRU for one decoding step from the last hidden state.
        """
       
        decoder_embed = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.decoder_gru(decoder_embed, last_hidden)

        sequence_length = encoder_output.shape[0]
        output_reshaped = decoder_output.repeat(sequence_length, 1, 1)
        
        # Bahdanau+ (2014): additive softmax 
        attention_scores = self.softmax(torch.tanh(self.attention(torch.cat((output_reshaped, encoder_output), dim=2))))
        
        # Apply mask to attention scores for padded tokens 
        masked_attention_scores = attention_scores * torch.unsqueeze(encoder_mask, -1)
        final_attention_scores = torch.transpose(masked_attention_scores, 0, 1)
        
        # Apply attention scores on encoder output to get context vectors
        attention_context = torch.bmm(torch.permute(encoder_output, (1,2,0)), final_attention_scores)
        attention_context_reshaped = self.l_reshape(torch.permute(attention_context, (2,0,1)))

        linear_concat = torch.cat((decoder_output, attention_context_reshaped), dim=2)

        logits = self.l_output(linear_concat)
        return logits[0], decoder_hidden, final_attention_scores