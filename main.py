# ============================================================================
# main.py - Main Execution Script
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloader
from models import ImageCaptioningModel
from train import train_model
from inference import generate_caption
from evaluate import evaluate_bleu


def main():
    """Main execution function"""
    
    # Configuration
    config = {
        'data_dir': './flickr8k/Images',  # Update this path
        'captions_file': './flickr8k/captions.txt',  # Update this path
        'batch_size': 32,
        'embed_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'device': 'mps'
    }
    
    print(f"Using device: {config['device']}")
    
    # Step 1: Load Data
    print("\n=== Loading Dataset ===")
    train_loader, dataset = get_dataloader(
        config['data_dir'],
        config['captions_file'],
        batch_size=config['batch_size']
    )
    
    vocab = dataset.vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of training samples: {len(dataset)}")
    
    # Step 2: Initialize Model
    print("\n=== Initializing Model ===")
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    ).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Step 3: Setup Training
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Step 4: Train
    print("\n=== Starting Training ===")
    model = train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        config['device'],
        num_epochs=config['num_epochs']
    )
    
    # Step 5: Evaluate
    print("\n=== Evaluating Model ===")
    bleu_score = evaluate_bleu(model, train_loader, vocab, config['device'], num_samples=500)
    
    # Step 6: Test Inference
    print("\n=== Testing Inference ===")
    test_image = './flickr8k/Images/667626_18933d713e.jpg'  # Update with actual image path
    if os.path.exists(test_image):
        caption = generate_caption(model, test_image, vocab, config['device'])
        print(f"Generated Caption: {caption}")
    
    print("\n=== Training Complete ===")
    print(f"Final BLEU-4 Score: {bleu_score:.4f}")
    print(f"Target: ≥ 0.25")
    print(f"Status: {'✓ ACHIEVED' if bleu_score >= 0.25 else '✗ NOT ACHIEVED'}")


if __name__ == "__main__":
    main()