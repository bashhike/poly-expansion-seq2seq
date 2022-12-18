#! /bin/python 

import argparse
import torch 
import torch.nn as nn 
from preprocessing import PolynomialDataset, Vocabulary, create_dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import pickle 

from models import Seq2seqBaseline
from utils import *

# Create te vocabulary on the complete dataset. 
with open('vocab.pkl', 'rb') as fp:
    vocab = pickle.load(fp)


def read_raw(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data 

def split_raw_lines(raw_data, delim='='):
    samples = []
    for line in raw_data.splitlines():
        samples.append(line.strip().split(delim))
    return samples 

def train(model, data_loader, num_epochs, model_file, learning_rate=0.0001):
    """Train the model for given number of epochs and save the trained model in 
    the final model_file.
    """

    decoder_learning_ratio = 5.0
    
    encoder_parameter_names = ['embedding', 'encoder_gru']
                               
    encoder_named_params = list(filter(lambda kv: any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    decoder_named_params = list(filter(lambda kv: not any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    encoder_params = [e[1] for e in encoder_named_params]
    decoder_params = [e[1] for e in decoder_named_params]
    optimizer = torch.optim.AdamW([{'params': encoder_params},
                {'params': decoder_params, 'lr': learning_rate * decoder_learning_ratio}], lr=learning_rate)
    
    clip = 50.0
    for epoch in trange(num_epochs, desc="training", unit="epoch"):
        # print(f"Total training instances = {len(train_dataset)}")
        # print(f"train_data_loader = {len(train_data_loader)} {1180 > len(train_data_loader)/20}")
        with tqdm(
                data_loader,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch_data in enumerate(batch_iterator, start=1):
                source, target = batch_data["example_tensors"]
                optimizer.zero_grad()
                loss = model.compute_loss(source, target)
                total_loss += loss.item()
                loss.backward()
                # Gradient clipping before taking the step
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item())
    # Save the model after training         
    torch.save(model.state_dict(), model_file)

######################################################################################
# Test the results on the validation data. 
######################################################################################

def predict_greedy(model, sentence, max_length=29):
    """Make predictions for the given input using greedy inference.
    """

    model.eval()

    # Convert sentence to the appropriate format. 
    word_ids = torch.unsqueeze(torch.LongTensor(vocab.get_ids_from_sentence(sentence)), dim=1).to(device)
    encoded_sequence, encoder_mask, encoder_hidden = model.encode(word_ids)

    cur_token = bos_id
    cur_hidden = encoder_hidden 
    res_tokens = [bos_id]
    
    # Iterate until eos is found or max sentence length is hit 
    while(cur_token != eos_id and len(res_tokens)<max_length):
        decoder_input = torch.unsqueeze(torch.LongTensor([cur_token]), dim=1).to(device)

        logits, decoder_hidden, attention_weights = model.decode(decoder_input, cur_hidden, encoded_sequence, encoder_mask)
        # Find the word with the highest probability 
        pred = torch.argmax(logits, dim=1)
        # Update current token and hidden state 
        cur_token = pred[0].item()
        cur_hidden = decoder_hidden 
        res_tokens.append(cur_token)
    return vocab.decode_sentence_from_ids(res_tokens)

def predict_beam(model, sentence, k=5, max_length=29):
    """Make predictions for the given inputs using beam search.
    """
    alpha = 0.7
    model.eval()
    
    # Convert sentence to the appropriate format. 
    word_ids = torch.unsqueeze(torch.LongTensor(vocab.get_ids_from_sentence(sentence)), dim=1).to(device)
    encoded_sequence, encoder_mask, encoder_hidden = model.encode(word_ids)

    # Fix the shape of encoder output to be able to run a batch 
    encoder_hidden = encoder_hidden.repeat(1,k,1) 
    encoded_sequence = encoded_sequence.repeat(1,k,1) 
    encoder_mask = encoder_mask.repeat(1,k)

    # Find the starting k candidates using the bos token 
    cur_decoder_input = torch.LongTensor([[bos_id]*k]).to(device)
    cur_probs, cur_hidden, _ = model.decode(cur_decoder_input, encoder_hidden, encoded_sequence, encoder_mask)
    cur_probs = cur_probs.log_softmax(-1)

    probs, indices = torch.topk(cur_probs[[0]], k, dim=1)
    
    vocab_size = cur_probs.shape[1]
    
    # Initialize variables for the search 
    candidates = indices
    next_chars = indices 
    best_candidates = torch.squeeze(indices)

    prev_probs = torch.transpose(probs, 0, 1)
    prev_hidden = cur_hidden

    # Store the finished candidates with their scores for re-ranking later. 
    sentences = []
    generations = []
    
    for i in range(max_length):
        # For each of the candidates, find the next tokens. 
        cur_probs, cur_hidden, _ = model.decode(next_chars, prev_hidden, encoded_sequence, encoder_mask)
        
        # Add prev probabilities 
        cur_probs = cur_probs.log_softmax(-1) + prev_probs.repeat(1,vocab_size) 
        
        probs, indices = torch.topk(cur_probs.flatten(), k, dim=0)
        prev_probs = probs.unsqueeze(-1)
        
        # Find the top k chars and the token that they came from 
        next_chars = torch.remainder(indices, vocab_size).unsqueeze(0)
        best_candidates = (indices/vocab_size).long()
        
        # Update the hidden state for next iteration 
        prev_hidden = cur_hidden[:, best_candidates, :]
        
        # Store candidates from beams 
        candidates = torch.cat([candidates[:, best_candidates], next_chars], dim=0)
        
        # If the current word is eos, then update the probability and add it to the sentences list 
        for w in range(k):
            if candidates[-1, w] == eos_id:
                generated_sentence = vocab.decode_sentence_from_ids(candidates[:, w].tolist())
                generated_sentence_len = len(candidates[:, w])
                generated_score = prev_probs[best_candidates[w], :]
                final_score = generated_score/(generated_sentence_len)**alpha

                sentences.append((final_score, generated_sentence))
                
                # Update the probability to remove the candidate from beam 
                prev_probs[best_candidates[w], :] += -1e9
        
        if len(sentences) >= k:
            break 

    # In case no eos token was found, build truncated sentences. 
    if len(sentences) == 0:
        for w in range(k):
            generated_sentence = vocab.decode_sentence_from_ids(candidates[:, w].tolist())
            generated_sentence_len = len(candidates[:, w])
            generated_score = prev_probs[best_candidates[w], :]
            final_score = generated_score/(generated_sentence_len)**alpha

            sentences.append((final_score, generated_sentence))

    # Sort by the final score and return 
    sentences.sort(reverse=True)
    for k,v in sentences[:5]:
        generations.append(v)
    return generations 

def run_training(vocab, checkpoint_file, num_epochs=10, batch_size=1536):
    train_raw = read_raw('train.txt')
    train_examples = split_raw_lines(train_raw)
    train_dataset = PolynomialDataset(train_examples, vocab)
    train_dataloader = create_dataloader(train_dataset, batch_size) 
    
    baseline_model = Seq2seqBaseline(vocab).to(device)
    train(baseline_model, train_dataloader, num_epochs, checkpoint_file, learning_rate=0.001)
    print("training complete!")



def run_testing(vocab, checkpoint_file): 
    baseline_model = Seq2seqBaseline(vocab).to(device)
    baseline_model.load_state_dict(torch.load(checkpoint_file, map_location=device))

    test_raw = read_raw('test.txt')
    test_examples = split_raw_lines(test_raw)

    # Loop over the test examples and check the accuracy 
    correct = 0 
    for src, tgt in test_examples:
        pred = predict_greedy(baseline_model, src)
        if src.strip() == tgt.strip():
            correct += 1
    print("Greedy Accuracy: {}".format(correct/len(test_examples))) 

    # Loop over the test examples and check the accuracy 
    # correct = 0 
    # for src, tgt in test_examples:
    #     pred = predict_beam(baseline_model, src)[0]
    #     if src.strip() == tgt.strip():
    #         correct += 1
    # print("Beam Accuracy: {}".format(correct/len(test_examples))) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Train or test the model.", 
        description="For testing both the trained model weights should be passed. For training, the model weights would be stored in the passed file."
    )
    parser.add_argument("--job", type=str, choices=['train', 'test'])
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--model_weights", type=str)
    args = parser.parse_args()
    
    # Create te vocabulary on the complete dataset. 
    with open(args.vocab_file, 'rb') as fp:
        vocab = pickle.load(fp)

    if args.job == 'train':
        run_training(vocab, args.model_weights) 
    else:
        run_testing(vocab, args.model_weights)