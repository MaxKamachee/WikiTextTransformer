import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import time
import math
import os
import json

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class WikiTextDataset(Dataset):
    def __init__(self, data, vocab, seq_length):
        self.data = [vocab.get(token, vocab['<UNK>']) for text in data for token in text.split()]
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_length]
        target_seq = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

def build_vocab(data, max_size=10000):
    counter = Counter()
    for text in data:
        counter.update(text.split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

def prepare_dataloaders(batch_size=64, seq_length=35, max_samples=100000):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    train_data = dataset['train']['text'][:max_samples]
    valid_data = dataset['validation']['text'][:max_samples//10]
    
    vocab = build_vocab(train_data)
    
    train_dataset = WikiTextDataset(train_data, vocab, seq_length)
    valid_dataset = WikiTextDataset(valid_data, vocab, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader, vocab

def train(model, train_loader, val_loader, vocab, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss / 200
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'lr {scheduler.get_last_lr()[0]:02.4f} | ms/batch {elapsed * 1000 / 200:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
                total_loss = 0
                start_time = time.time()

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:5.2f} | '
              f'valid ppl {math.exp(val_loss):8.2f}')
        scheduler.step()

    # Save the model and vocabulary
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
    }, save_path)
    print(f"Model and vocabulary saved to {save_path}")

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            total_loss += criterion(output.view(-1, output.size(-1)), targets.view(-1)).item()
    return total_loss / len(data_loader)

def generate_text(model, vocab, start_text, max_length=50, temperature=0.7):
    model.eval()
    reverse_vocab = {v: k for k, v in vocab.items()}
    tokens = [vocab.get(word, vocab['<UNK>']) for word in start_text.split()]
    generated = tokens.copy()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).item()
            generated.append(next_token)
            tokens.append(next_token)
            
            if next_token == vocab['<PAD>']:
                break
    
    return ' '.join([reverse_vocab.get(token, '<UNK>') for token in generated])

def save_hyperparameters(hyperparameters, file_path):
    with open(file_path, 'w') as f:
        json.dump(hyperparameters, f)

def load_hyperparameters(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Hyperparameters
        hyperparameters = {
            'batch_size': 32,
            'seq_length': 35,
            'd_model': 128,
            'nhead': 2,
            'd_hid': 256,
            'nlayers': 2,
            'dropout': 0.2,
            'epochs': 3,
            'lr': 0.001,
            'max_samples': 30000
        }

        save_path = 'wikitext_small_transformer.pth'
        hyperparameters_path = 'hyperparameters.json'

        should_train = True
        if os.path.exists(save_path) and os.path.exists(hyperparameters_path):
            print("Loading pre-trained model...")
            try:
                checkpoint = torch.load(save_path)
                saved_hyperparameters = load_hyperparameters(hyperparameters_path)
                
                if saved_hyperparameters == hyperparameters:
                    vocab = checkpoint['vocab']
                    model = TransformerModel(len(vocab), hyperparameters['d_model'], 
                                             hyperparameters['nhead'], hyperparameters['d_hid'], 
                                             hyperparameters['nlayers'], hyperparameters['dropout']).to(device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Model loaded successfully.")
                    should_train = False
                else:
                    print("Hyperparameters have changed. Retraining the model.")
            except Exception as e:
                print(f"Error loading pre-trained model: {str(e)}")
                print("Falling back to training a new model.")

        if should_train:
            print("Preparing data...")
            train_loader, val_loader, vocab = prepare_dataloaders(
                hyperparameters['batch_size'], 
                hyperparameters['seq_length'], 
                hyperparameters['max_samples']
            )
            
            print("Initializing model...")
            model = TransformerModel(len(vocab), hyperparameters['d_model'], 
                                     hyperparameters['nhead'], hyperparameters['d_hid'], 
                                     hyperparameters['nlayers'], hyperparameters['dropout']).to(device)
            
            print("Starting training...")
            start_time = time.time()
            train(model, train_loader, val_loader, vocab, hyperparameters['epochs'], 
                  hyperparameters['lr'], device, save_path)
            total_time = time.time() - start_time
            print(f"Total training time: {total_time:.2f} seconds")

            # Save hyperparameters
            save_hyperparameters(hyperparameters, hyperparameters_path)

        # Generate some text
        while True:
            start_text = input("Enter a prompt (or 'q' to quit): ")
            if start_text.lower() == 'q':
                break
            generated_text = generate_text(model, vocab, start_text, max_length=50)
            print(f"Generated text:\n{generated_text}\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()