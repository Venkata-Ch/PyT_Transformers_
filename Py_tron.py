import logging
import torch
import torch.nn as nn
import pickle
from torch.nn import Transformer


class Attention(nn.Module):
    def __init__(self,embedding_size, head_count):
        try:
            super(Attention, self).__init__()
            self.embedding_size = embedding_size
            self.head_count = head_count
            self.head_dimensions = embedding_size // head_count
            # assert (self.head_dimensions * head_count == embedding_size), 'Embedding size must be divided by head_count'

            self.vals = nn.Linear(self.head_dimensions, self.head_dimensions,bias=False)
            self.keys = nn.Linear(self.head_dimensions,self.head_dimensions,bias=False)
            self.queries = nn.Linear(self.head_dimensions,self.head_dimensions, bias=False)
            self.forward_out = nn.Linear(head_count*self.head_dimensions, embedding_size)


        except Exception as error:
            logging.error(str(error))

    def forward(self,keys,query,values,mask):
        N = query.shape[0]
        value_length,key_length,query_length = values.shape[1], keys.shape[1],query.shape[1]
        values = values.reshape(N, value_length, self.head_dimensions,self.head_count)
        keys = keys.reshape(N, key_length,self.head_dimensions,self.head_count)
        query = query.reshape(N,query_length,self.head_dimensions,self.head_count)

        ene = torch.einsum("nqhd,nkhd->nhqk", [query,keys])

        if mask is not None:
            ene = ene.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(ene / (self.embedding_size ** (1/2)), dim=3)
        output = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(N,query_length,self.head_count*self.head_dimensions)
        output = self.forward_out(output)
        return output


class Transformer_Block(nn.Module):
    def __init__(self, embedding_size,head_count,drop_out,forwardexpansion):
        super(Transformer_Block, self).__init__()
        self.attention = Attention(embedding_size,head_count)
        self.normalization1 = nn.LayerNorm(embedding_size)
        self.normalization2 = nn.LayerNorm(embedding_size)
        print('&&&&&&&&&&&&&&&&',head_count)
        print('*****************',embedding_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, forwardexpansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forwardexpansion * embedding_size,embedding_size)
        )
        self.drop_out = nn.Dropout(drop_out)

    def forward(self,value,key,query,mask):
        attention = self.attention(value,query,key,mask)
        a = self.drop_out(self.normalization1(attention + query))
        forward_ = self.feedforward(a)
        return forward_

class Encoder(nn.Module):
    def __init__(self,src_vocabsize,embedding_size,
                 head_count,device,forwardexpansion,
                 layer_num, drop_out,max_length):

        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.wrd_embedding = nn.Embedding(src_vocabsize,embedding_size)
        self.position_embedder = nn.Embedding(max_length, embedding_size)
        print("###########################",embedding_size)
        print("$$$$$$$$$$$$$$$$$$$",head_count)
        self.layers = nn.ModuleList(
            [
                Transformer_Block(embedding_size,
                            head_count,
                            drop_out=drop_out,
                            forwardexpansion=forwardexpansion,)
            ]
        )
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x, mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, sequence_length).to(self.device)
        out = self.drop_out(self.wrd_embedding(x) + self.position_embedder(positions) )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class Decoder_(nn.Module):
    def __init__(self,embedding_size,head_count,forwardexpansion,drop_out,device):
        super(Decoder_, self).__init__()

        self.attention = Attention(embedding_size,head_count)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_ = Transformer_Block(embedding_size, head_count,drop_out,forwardexpansion)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self,x,value,key,srcmask,trgmask):
        attention = self.attention(x, x, x, trgmask)
        query = self.drop_out(self.norm(attention + x))
        out = self.transformer_(value, key, query, srcmask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 tr_vocabsize,
                 embedding_size,
                 layer_num,
                 head_count,
                 forwardexpansion,
                 drop_out,device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.wrd_embedding = nn.Embedding(tr_vocabsize,embedding_size)
        self.position_embed = nn.Embedding(max_length, embedding_size)

        self.layers = nn.ModuleList(
            [Decoder_(embedding_size,head_count, forwardexpansion,drop_out, device)
             for _ in range(layer_num)]
        )
        self.f_out = nn.Linear(embedding_size,tr_vocabsize)
        self.dropout = nn.Dropout(drop_out)


    def forward(self, x, enc_output,src_mask,tr_mask):
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, sequence_length).to(self.device)
        x = self.dropout((self.wrd_embedding(x) + self.position_embed(positions)))

        for layer in self.layers:
            x = layer(x, enc_output, enc_output, src_mask, tr_mask)
        out = self.f_out(x)
        return out



class Transformer(nn.Module):
    def __init__(self,src_vocabsize, tr_vocabsize,src_padidx,tr_padidx,embedding_size=256,forwardexpansion=4,
               layer_num=6,head_count = 8, drop_out=0,
               device = "cpu",
               max_length=100
               ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocabsize,embedding_size,
                               head_count,device,forwardexpansion,layer_num,drop_out,max_length)
        self.decoder = Decoder(
            tr_vocabsize,
            embedding_size,
            layer_num, head_count,
            forwardexpansion,
            drop_out,
            device,
            max_length
        )
        self.src_padidx = src_padidx
        self.tr_padidx = tr_padidx
        self.device = device

    def src_make_masking(self, src):
        src_mask = (src != self.src_padidx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def tr_make_masking(self, trg):
        N, tr_len = trg.shape
        trg_mask = torch.tril(torch.ones((tr_len, tr_len))).expand(N, 1, tr_len,
                                                                   tr_len)
        return trg_mask.to(self.device)

    def forward(self, src,trg):
        src_masker = self.src_make_masking(src)
        trg_masker = self.tr_make_masking(trg)
        src_enc = self.encoder(src, src_masker)
        out = self.decoder(trg, src_enc, src_masker, trg_masker)
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 8], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_padidx = 0
    tr_padidx = 0
    src_vocabsize = 10
    tr_vocabsize = 10
    model = Transformer(src_vocabsize, tr_vocabsize, src_padidx, tr_padidx).to(device)
    out = model(x, trg[:, :-1])
    print(out)
    with open('models/tranformer.pt', 'wb+') as f:
        pickle.dump(model, f)
    return out
if __name__=="__main__":
    main()









