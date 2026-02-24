# ============================================================================
# models.py - Vision Encoder + Transformer Decoder
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
import math


class VisionEncoder(nn.Module):
    """Vision Encoder using ResNet-50"""
    def __init__(self, embed_size=512):
        super(VisionEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        # Remove final FC layer
        modules = list(resnet.children())[:-2]  # Output: [B, 2048, 7, 7]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Projection to match decoder dimension
        self.projection = nn.Linear(2048, embed_size)
        
    def forward(self, images):
        # images: [B, 3, 224, 224]
        features = self.resnet(images)  # [B, 2048, 7, 7]
        features = features.permute(0, 2, 3, 1)  # [B, 7, 7, 2048]
        features = features.view(features.size(0), -1, features.size(-1))  # [B, 49, 2048]
        features = self.projection(features)  # [B, 49, embed_size]
        return features


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerDecoder(nn.Module):
    """Transformer Decoder for Caption Generation"""
    def __init__(self, vocab_size, embed_size=512, num_layers=6, num_heads=8, 
                 forward_expansion=4, dropout=0.1, max_len=100):
        super(TransformerDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, captions, encoder_out, tgt_mask=None, tgt_padding_mask=None):
        # captions: [B, seq_len]
        # encoder_out: [B, 49, embed_size]
        
        embedded = self.dropout(self.word_embedding(captions))  # [B, seq_len, embed_size]
        embedded = self.position_encoding(embedded)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        logits = self.fc_out(output)  # [B, seq_len, vocab_size]
        return logits
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    def __init__(self, vocab_size, embed_size=512, num_layers=6, num_heads=8, dropout=0.1):
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = VisionEncoder(embed_size=embed_size)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, images, captions, tgt_padding_mask=None):
        # Encode images
        encoder_out = self.encoder(images)  # [B, 49, embed_size]
        
        # Generate causal mask
        tgt_len = captions.size(1)
        tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len).to(captions.device)
        
        # Decode
        logits = self.decoder(captions, encoder_out, tgt_mask=tgt_mask, 
                            tgt_padding_mask=tgt_padding_mask)
        
        return logits