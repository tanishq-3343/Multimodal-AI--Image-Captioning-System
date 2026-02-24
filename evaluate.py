# ============================================================================
# evaluate.py - BLEU Score Calculation
# ============================================================================

from nltk.translate.bleu_score import corpus_bleu
import torch


def evaluate_bleu(model, dataloader, vocab, device, num_samples=1000):
    """Calculate BLEU-4 score on dataset"""
    model.eval()
    
    references = []
    hypotheses = []
    
    print("Generating captions for evaluation...")
    
    with torch.no_grad():
        for i, (images, captions) in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
                
            images = images.to(device)
            
            # Generate captions
            for j in range(images.size(0)):
                img = images[j:j+1]
                encoder_out = model.encoder(img)
                
                # Generate caption
                caption = [vocab.stoi["<START>"]]
                for _ in range(50):
                    caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
                    tgt_mask = model.decoder.generate_square_subsequent_mask(len(caption)).to(device)
                    
                    output = model.decoder(caption_tensor, encoder_out, tgt_mask=tgt_mask)
                    next_word = output[0, -1, :].argmax().item()
                    
                    caption.append(next_word)
                    if next_word == vocab.stoi["<END>"]:
                        break
                
                # Convert to words
                hypothesis = [vocab.itos[idx] for idx in caption[1:-1]]
                
                # Reference caption
                ref_caption = captions[j].cpu().numpy()
                reference = [[vocab.itos[idx] for idx in ref_caption if idx not in [0, 1, 2, 3]]]
                
                hypotheses.append(hypothesis)
                references.append(reference)
    
    # Calculate BLEU-4
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"\nBLEU-4 Score: {bleu4:.4f}")
    
    return bleu4