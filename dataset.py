# ============================================================================
# dataset.py - Data Loading and Preprocessing
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in nltk.tokenize.word_tokenize(sentence.lower()):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        tokenized = nltk.tokenize.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized]


class Flickr8kDataset(Dataset):
    """
    Flickr8k Dataset Loader
    Download from: https://www.kaggle.com/datasets/adityajn105/flickr8k
    
    Structure:
    flickr8k/
    ├── Images/
    └── captions.txt
    """
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.captions = []
        
        # Load captions
        with open(captions_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    self.imgs.append(img_name)
                    self.captions.append(caption)
        
        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        # Numericalize caption
        numericalized = [self.vocab.stoi["<START>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi["<END>"])
        
        return img, torch.tensor(numericalized)


class CaptionCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, targets


def get_dataloader(root_dir, captions_file, batch_size=32, shuffle=True, num_workers=2):
    """Create dataloader with proper transforms"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDataset(root_dir, captions_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=CaptionCollate(pad_idx=pad_idx)
    )
    
    return loader, dataset