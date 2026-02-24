# ============================================================================
# inference.py - Caption Generation
# ============================================================================

import torch
from PIL import Image
from torchvision import transforms


def generate_caption(model, image_path, vocab, device, max_len=50, method='greedy'):
    """Generate caption for a single image"""
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Encode image
    with torch.no_grad():
        encoder_out = model.encoder(image)
    
    # Start with <START> token
    caption = [vocab.stoi["<START>"]]
    
    for _ in range(max_len):
        caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Create mask
            tgt_mask = model.decoder.generate_square_subsequent_mask(len(caption)).to(device)
            
            # Predict next word
            output = model.decoder(caption_tensor, encoder_out, tgt_mask=tgt_mask)
            
            # Get last word predictions
            pred = output[0, -1, :]
            
            if method == 'greedy':
                next_word = pred.argmax().item()
            else:  # sampling
                probs = torch.softmax(pred, dim=0)
                next_word = torch.multinomial(probs, 1).item()
        
        caption.append(next_word)
        
        # Stop if <END> token
        if next_word == vocab.stoi["<END>"]:
            break
    
    # Convert to words
    caption_words = [vocab.itos[idx] for idx in caption[1:-1]]  # Remove <START> and <END>
    return ' '.join(caption_words)