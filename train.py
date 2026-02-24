# ============================================================================
# train.py - Training Script
# ============================================================================

import torch
import torch.nn as nn
from tqdm import tqdm
import os


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=20, 
                save_path='checkpoints'):
    """Training loop with progress tracking"""
    
    os.makedirs(save_path, exist_ok=True)
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, captions) in enumerate(progress_bar):
            images = images.to(device)
            captions = captions.to(device)
            
            # Teacher forcing: input is caption[:-1], target is caption[1:]
            inp = captions[:, :-1]
            target = captions[:, 1:]
            
            # Create padding mask
            padding_mask = (inp == 0)
            
            # Forward pass
            outputs = model(images, inp, tgt_padding_mask=padding_mask)
            
            # Compute loss
            outputs = outputs.reshape(-1, outputs.shape[2])
            target = target.reshape(-1)
            
            loss = criterion(outputs, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, 'best_model.pth'))
            print(f'Best model saved with loss: {avg_loss:.4f}')
    
    return model