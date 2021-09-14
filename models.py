import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(64, max_len, d_model).to(device)
        pe[:,:, 0::2] = torch.sin(position * div_term)
        pe[:,:, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0),:x.size(1),:]
        return x

class resnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.ffn_1 = nn.Linear(60*54, 1024)

        self.ffn_11 = nn.Linear(1024, 512)
        self.ffn_12 = nn.Linear(512, 1024)

        self.ffn_2 = nn.Linear(1024, 512)

        self.ffn_21 = nn.Linear(512, 256)
        self.ffn_22 = nn.Linear(256, 512)

        self.ffn_3 = nn.Linear(512, 1)
        
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        

    def forward(self, feature):
        x = self.flatten(feature)
        
        x = self.ffn_1(x)
        res = F.leaky_relu(self.ffn_11(x))
        res = self.ffn_12(res)
        x = F.leaky_relu(x+res)
        x = self.dropout_1(x)
        
        x = self.ffn_2(x)
        res = F.leaky_relu(self.ffn_21(x))
        res = self.ffn_22(res)
        x = F.leaky_relu(x+res)
        x = self.dropout_2(x)
        
        x = self.ffn_3(x)
        
        return x
    
class transformerModel(nn.Module):
    def __init__(self, embed_dim=256, enc_num=1, num_heads=16, 
                ff_dim=512, rate=0.5, in_num=1):
        super().__init__()
        self.pos_enc = PositionalEncoding(54)
        
        encoder_layers = nn.TransformerEncoderLayer(
            embed_dim, num_heads, ff_dim, rate, batch_first=True)
        decoder_layers = nn.TransformerDecoderLayer(
            embed_dim, num_heads, ff_dim, rate, batch_first=True)
        
        self.self_encoders = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layers, in_num) for idx in range(enc_num)])
        self.cross_encoders = nn.ModuleList(
            [nn.TransformerDecoder(decoder_layers, in_num) for idx in range(enc_num)])
                
        self.ffn_token_embedding_1 = nn.Linear(60*54, 512)
        self.ffn_token_embedding_2 = nn.Linear(512, embed_dim)
        self.ffn_embedding = nn.Linear(54, embed_dim)
        
        self.ffn_decoder_1 = nn.Linear(embed_dim, 512)
        self.ffn_decoder_2 = nn.Linear(512, 128)
        self.dropout_1 = nn.Dropout(rate)
        self.dropout_2 = nn.Dropout(rate)
        
        self.ffn_logit = nn.Linear(128, 1)
        self.flt_1 = nn.Flatten()
        self.flt_2 = nn.Flatten()
        
    def forward(self, feature):
        
        x = self.pos_enc(feature)
        x = self.ffn_embedding(x)
        
        token_in = self.ffn_token_embedding_1(self.flt_1(feature))
        token_in = F.leaky_relu(token_in)
        token = self.ffn_token_embedding_2(token_in)
        token = torch.reshape(token, (token.shape[0],1,-1))
        
        for self_enc, cross_enc in zip(self.self_encoders, self.cross_encoders):
            x = self_enc(x) 
            token = cross_enc(token,x) 

        token = self.flt_2(token)
        token = self.ffn_decoder_1(token)
        token = F.leaky_relu(token)
        token = self.dropout_1(token)
        token = F.leaky_relu(self.ffn_decoder_2(token))
        token = self.dropout_2(token)
        
        token = self.ffn_logit(token)
        
        
        return token

class ensembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_1 = resnetModel()
        self.model_2 = transformerModel()

        self.ffn_1 = nn.Linear(2,256)
        self.dropout_1 = nn.Dropout(0.5)
        self.ffn_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.ffn_3 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = self.model_1(x)
        x2 = self.model_2(x)
        
        x = torch.cat([x1,x2],axis=1)
        x = self.dropout_1(F.leaky_relu(self.ffn_1(x)))
        x = self.dropout_2(F.leaky_relu(self.ffn_2(x)))
        x = self.ffn_3(x)
            
        return x