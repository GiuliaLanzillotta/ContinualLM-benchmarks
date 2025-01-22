#https://colab.research.google.com/#scrollTo=4ce6f4c3-41b2-4610-a68e-fef96c8fe761&fileId=https%3A//huggingface.co/datasets/bird-of-paradise/transformer-from-scratch-tutorial/blob/main/Transformer_Implementation_Tutorial.ipynb

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math

def embedding_layer_sampling(x:torch.Tensor, embedding:nn.Module, embedding_var:Parameter=None, num_samples=1, device='cuda'):
    """ Sample from an embedding layer with mean and variance through the reparametrization trick."""
    mean = embedding(x) #[2, 16, 1024]     
    std = embedding_var(x)

    eps = torch.empty(mean.shape, device=device).normal_(mean=0,std=1)
    output = mean + (eps * std)
    return output


def linear_layer_sampling(x:torch.Tensor, layer_fun:nn.Module, weight_var:Parameter, bias_var:Parameter, num_samples=1, device='cuda'):
    """ Sample from a linear layer with mean and variance through the reparametrization trick."""
    mean = layer_fun(x) #[2, 16, 1024]     
    std = torch.sqrt((x**2).matmul(torch.exp(weight_var.t())) \
                     + torch.exp(bias_var).unsqueeze(0).unsqueeze(0))
        
    eps = torch.empty(mean.shape, device=device).normal_(mean=0,std=1)
    output = mean + (eps * std)
    return output


class VariationalLayer(nn.Module):
    def __init__(self, prior_var=1.0):
        super().__init__()
        self.prior_var = prior_var

    def init_variational_params(self, layer, bias=True):

        weight_var = Parameter(torch.Tensor(layer.weight.size()))
        weight_prior_mean = Parameter(torch.Tensor(layer.weight.size()))
        weight_prior_var = Parameter(torch.Tensor(layer.weight.size())) * np.log(self.prior_var)

        if bias: 
            bias_var = Parameter(torch.Tensor(layer.bias.size()))
            bias_prior_mean = Parameter(torch.Tensor(layer.bias.size()))
            bias_prior_var = Parameter(torch.Tensor(layer.bias.size())) * np.log(self.prior_var)
            nn.init.constant_(bias_var, bias_prior_var)
            
        else: 
            bias_var = 0
            bias_prior_mean = 0
            bias_prior_var = 0


        nn.init.constant_(weight_var, weight_prior_var)

        return weight_var, bias_var, weight_prior_mean, bias_prior_mean, weight_prior_var, bias_prior_var

    @staticmethod
    def init_weights(module, mean=0.0, std=1.0):
            if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def reset_parameters(self):
        self.apply(self.init_weights)

    
    def to_device(self, device):
        for name, param in self.named_parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)
        self.to(device)

class VariationalLinearLayer(VariationalLayer):
    def __init__(self, in_features, out_features, bias=True, prior_var=1.0, device='cuda'):
        super().__init__(prior_var)
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.bias = bias

        self.linear = nn.Linear(in_features, out_features)
        self.weight_var, self.bias_var, self.weight_prior_mean, self.bias_prior_mean, self.weight_prior_var, self.bias_prior_var = self.init_variational_params(self.linear, bias=bias)

        self.reset_parameters()
        self.to(device)

    def forward(self, x):
        return linear_layer_sampling(x, self.linear, self.weight_var, self.bias_var, device=self.device)

class VariationalEmbeddingWithProjection(VariationalLayer):
    def __init__(self, vocab_size, d_embed, d_model,  
                 max_position_embeddings =512, dropout=0.1, prior_var = 1.0, device = 'cuda'):
        super().__init__(prior_var)
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.projection = VariationalLinearLayer(self.d_embed, self.d_model, bias=False, prior_var=prior_var, device=device)
        self.scaling = float(math.sqrt(self.d_model))
        self.device = device

        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.embedding_var = nn.Embedding(self.vocab_size, self.d_embed)
        self.embedding_prior_mean = Parameter(torch.Tensor(self.embedding.weight.size()))
        self.embedding_prior_var = Parameter(torch.Tensor(self.embedding.weight.size())) * np.log(self.prior_var)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.constant_(self.embedding_var.weight, self.prior_var)

    @staticmethod
    def create_positional_encoding(seq_length, d_model, batch_size=1):
        # Create position indices: [seq_length, 1]
        position = torch.arange(seq_length).unsqueeze(1).float()
        
        # Create dimension indices: [1, d_model//2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Create empty tensor: [seq_length, d_model]
        pe = torch.zeros(seq_length, d_model)
        
        # Compute sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and expand: [batch_size, seq_length, d_model]
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe


    def forward(self, x):
        assert x.dtype == torch.long, f"Input tensor must have dtype torch.long, got {x.dtype}"
        batch_size, seq_length = x.size() # [batch, seq_length]
        embedding_output = embedding_layer_sampling(x,self.embedding, self.embedding_var, device=self.device) #[2, 16, 1024]     

        projection_output = self.projection(embedding_output) #[2, 16, 768]
        projection_output = projection_output * self.scaling  

        # add positional encodings to projected, 
        # scaled embeddings before applying layer norm and dropout.
        positional_encoding = self.create_positional_encoding(seq_length, self.d_model, batch_size)    #[2, 16, 768]
        
        # In addition, we apply dropout to the sums of the embeddings 
        # in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.
        normalized_sum = self.layernorm(projection_output + positional_encoding)
        final_output = self.dropout(normalized_sum)
        return final_output


class VariationalTransformerAttention(VariationalLayer):
    """
    Transformer Scaled Dot Product Attention Module
    Args:
        d_model: Total dimension of the model.
        num_head: Number of attention heads.
        dropout: Dropout rate for attention scores.
        bias: Whether to include bias in linear projections.

    Inputs:
        sequence: input sequence for self-attention and the query for cross-attention
        key_value_state: input for the key, values for cross-attention
    """
    def __init__(self, d_model, num_head, dropout=0.1, bias=True, prior_var = 1.0, device = 'cuda'): # infer d_k, d_v, d_q from d_model
        super().__init__(prior_var)
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        self.d_model = d_model
        self.num_head = num_head
        self.d_head=d_model//num_head
        self.dropout_rate = dropout  # Store dropout rate separately
        self.device = device

        # linear transformations
        self.q_proj = VariationalLinearLayer(d_model, d_model, bias=bias, prior_var=prior_var, device=device)
        self.k_proj = VariationalLinearLayer(d_model, d_model, bias=bias, prior_var=prior_var, device=device)
        self.v_proj = VariationalLinearLayer(d_model, d_model, bias=bias, prior_var=prior_var, device=device)
        self.output_proj = VariationalLinearLayer(d_model, d_model, bias=bias, prior_var=prior_var, device=device)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head)) # Store as float in initialization
        

    def forward(self, sequence, key_value_states = None, att_mask=None):
        """Input shape: [batch_size, seq_len, d_model=num_head * d_head]"""
        batch_size, seq_len, model_dim = sequence.size()

        # Check only critical input dimensions
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
            f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"


        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        
        # Linear projections and reshape for multi-head
        Q_state = self.q_proj(sequence) # [batch_size, seq_len, d_model]
        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_len
            K_state = self.k_proj(sequence)
            V_state = self.v_proj(sequence)

        #[batch_size, self.num_head, seq_len, self.d_head]
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2) 
            
        # in cross-attention, key/value sequence length might be different from query sequence length
        K_state = K_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)

        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler
    
    
        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2)) 

    
        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask
        
        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim = -1)
    
        # apply drop out to attention score
        att_score = self.dropout(att_score)
    
        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)
    
        # concatinate all attention heads
        att_output = att_output.contiguous().view(batch_size, seq_len, self.num_head*self.d_head) 
    
        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), \
        f"Final output shape {att_output.size()} incorrect"

        return att_output
    

class VariationalFFN(VariationalLayer):
    """
    Position-wise Feed-Forward Networks
    This consists of two linear transformations with a ReLU activation in between.
    
    FFN(x) = max(0, xW1 + b1 )W2 + b2
    d_model: embedding dimension (e.g., 512)
    d_ff: feed-forward dimension (e.g., 2048)
    
    """
    def __init__(self, d_model, d_ff, prior_var=1.0, device='cuda'):
        super().__init__(prior_var)
        self.d_model=d_model
        self.d_ff= d_ff
        
        # Linear transformation y = xW+b
        self.fc1 = VariationalLinearLayer(self.d_model, self.d_ff, bias = True, prior_var=prior_var, device=device)
        self.fc2 = VariationalLinearLayer(self.d_ff, self.d_model, bias = True, prior_var=prior_var, device=device)



    def forward(self, input):
        # check input and first FF layer dimension matching
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"

        # First linear transformation followed by ReLU
        # There's no need for explicit torch.max() as F.relu() already implements max(0,x)
        f1 = F.relu(self.fc1(input)) # max(0, xW_1 + b_1)

        # max(0, xW_1 + b_1)W_2 + b_2 
        f2 =  self.fc2(f1) 

        return f2


class VariationalTransformerEncoder(nn.Module):
    """
    Encoder layer of the Transformer
    Sublayers: TransformerAttention
               Residual LayerNorm
               FNN
               Residual LayerNorm
    Args:
            d_model: 512 model hidden dimension
            d_embed: 512 embedding dimension, same as d_model in transformer framework
            d_ff: 2048 hidden dimension of the feed forward network
            num_head: 8 Number of attention heads.
            dropout:  0.1 dropout rate 
            
            bias: Whether to include bias in linear projections.
              
    """

    def __init__(
        self, d_model=512, d_embed=512, d_ff=2048, num_head=8, dropout=0.1, bias=True, device='cuda', prior_var=1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device

        
        # attention sublayer
        self.att = VariationalTransformerAttention(
            d_model = d_model,
            num_head = num_head,
            dropout = dropout,
            bias = bias,
            prior_var=prior_var,
            device=device
        )
        
        # FFN sublayer
        self.ffn = VariationalFFN(
            d_model = d_model,
            d_ff = d_ff,
            prior_var=prior_var,
            device=device
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # layer-normalization layer
        self.LayerNorm_att = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)

        
    def forward(self, embed_input, padding_mask=None):
       
        batch_size, seq_len, _ = embed_input.size()
        
        ## First sublayer: self attion 
        att_sublayer = self.att(sequence = embed_input, key_value_states = None, 
                                att_mask = padding_mask)  # [batch_size, sequence_length, d_model]
        
        # apply dropout before layer normalization for each sublayer
        att_sublayer = self.dropout(att_sublayer)
        # Residual layer normalization
        att_normalized = self.LayerNorm_att(embed_input + att_sublayer)  # [batch_size, sequence_length, d_model]
        
        ## Second sublayer: FFN
        ffn_sublayer = self.ffn(att_normalized)             # [batch_size, sequence_length, d_model]
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized + ffn_sublayer )   # [batch_size, sequence_length, d_model]
    

        return ffn_normalized


class VariationalTransformerDecoder(nn.Module):
    """
    Decoder layer of the Transformer
    Sublayers: TransformerAttention with self-attention
               Residual LayerNorm
               TransformerAttention with cross-attention
               Residual LayerNorm
               FNN
               Residual LayerNorm
    Args:
            d_model: 512 model hidden dimension
            d_embed: 512 embedding dimension, same as d_model in transformer framework
            d_ff: 2048 hidden dimension of the feed forward network
            num_head: 8 Number of attention heads.
            dropout:  0.1 dropout rate 
            
            bias: Whether to include bias in linear projections.
              
    """

    def __init__(
        self, d_model=512, d_embed=512, d_ff=2048, num_head=8, dropout=0.1, bias=True, device='cuda', prior_var=1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device

        
        # attention sublayer
        self.att = VariationalTransformerAttention(
            d_model = d_model,
            num_head = num_head,
            dropout = dropout,
            bias = bias,
            prior_var=prior_var,
            device=device
        )
        
        # FFN sublayer
        self.ffn = VariationalFFN(
            d_model = d_model,
            d_ff = d_ff,
            prior_var=prior_var,
            device=device
        )

        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # layer-normalization layer
        self.LayerNorm_att1 = nn.LayerNorm(self.d_model)
        self.LayerNorm_att2 = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)

    @staticmethod
    def create_causal_mask(seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    
    def forward(self, embed_input, cross_input, padding_mask=None):
        """
        Args:
        embed_input: Decoder input sequence [batch_size, seq_len, d_model]
        cross_input: Encoder output sequence [batch_size, encoder_seq_len, d_model]
        casual_attention_mask: Causal mask for self-attention [batch_size, seq_len, seq_len]
        padding_mask: Padding mask for cross-attention [batch_size, seq_len, encoder_seq_len]
        Returns:
        Tensor: Decoded output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = embed_input.size()
        
        assert embed_input.size(-1) == self.d_model, f"Input dimension {embed_input.size(-1)} doesn't match model dimension {self.d_model}"
        assert cross_input.size(-1) == self.d_model, "Encoder output dimension doesn't match model dimension"


        # Generate and expand causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_len).to(self.device)  # [seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]


        ## First sublayer: self attion 
        # After embedding and positional encoding, input sequence feed into current attention sublayer
        # Or, the output of the previous encoder/decoder feed into current attention sublayer
        att_sublayer1 = self.att(sequence = embed_input, key_value_states = None, 
                                att_mask = causal_mask)  # [batch_size, num_head, sequence_length, d_model]
        # apply dropout before layer normalization for each sublayer
        att_sublayer1 = self.dropout(att_sublayer1)
        # Residual layer normalization
        att_normalized1 = self.LayerNorm_att1(embed_input + att_sublayer1)           # [batch_size, sequence_length, d_model]

        ## Second sublayer: cross attention
        # Query from the output of previous attention output, or training data
        # Key, Value from output of Encoder of the same layer
        att_sublayer2 = self.att(sequence = att_normalized1, key_value_states = cross_input, 
                                att_mask = padding_mask)  # [batch_size, sequence_length, d_model]
        # apply dropout before layer normalization for each sublayer
        att_sublayer2 = self.dropout(att_sublayer2)
        # Residual layer normalization
        att_normalized2 = self.LayerNorm_att2(att_normalized1 + att_sublayer2)           # [batch_size, sequence_length, d_model]
        
        
        # Third sublayer: FFN
        ffn_sublayer = self.ffn(att_normalized2)                                   # [batch_size, sequence_length, d_model]
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized2 + ffn_sublayer )       # [batch_size, sequence_length, d_model]
    

        return ffn_normalized


class VariationalTransformerEncoderDecoder(nn.Module):
    """
    Encoder-Decoder stack of the Transformer
    Sublayers:  Encoder x 6
                Decoder x 6
    Args:
            d_model: 512 model hidden dimension
            d_embed: 512 embedding dimension, same as d_model in transformer framework
            d_ff: 2048 hidden dimension of the feed forward network
            num_head: 8 Number of attention heads.
            dropout:  0.1 dropout rate 
            
            bias: Whether to include bias in linear projections.
              
    """
    def __init__(
        self, num_layer, d_model=512, d_embed=512, d_ff=2048, num_head=8, dropout=0.1, bias=True, prior_var=1.0, device='cuda'
    ):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_ff = d_ff
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias
        self.device = device

        
        # Encoder stack
        self.encoder_stack = nn.ModuleList([VariationalTransformerEncoder(
                                        d_model = self.d_model, 
                                        d_embed = self.d_embed,
                                        d_ff = self.d_ff,
                                        num_head = self.num_head, 
                                        dropout = self.dropout,
                                        bias = self.bias, 
                                        prior_var=prior_var, 
                                        device=device) for _ in range(self.num_layer)])

        # Decoder stack
        self.decoder_stack = nn.ModuleList([VariationalTransformerDecoder(
                                        d_model = self.d_model, 
                                        d_embed = self.d_embed,
                                        d_ff = self.d_ff,
                                        num_head = self.num_head, 
                                        dropout = self.dropout,
                                        bias = self.bias, 
                                        prior_var=prior_var,
                                        device=device) for _ in range(self.num_layer)])

    
    def forward(self, embed_encoder_input, embed_decoder_input, padding_mask=None):
        # First layer of the encoder, decoder deck takes input from outside the deck
        encoder_output = embed_encoder_input
        decoder_output = embed_decoder_input

        for (encoder, decoder) in zip(self.encoder_stack, self.decoder_stack):
            encoder_output = encoder(embed_input = encoder_output, padding_mask = padding_mask)
            decoder_output = decoder(embed_input = decoder_output, cross_input =encoder_output, padding_mask=padding_mask)
       
        
        return decoder_output


class VariationalTransformer(nn.Module):
    def __init__(
        self, 
        num_layer,
        d_model, 
        d_embed, 
        d_ff,
        num_head,
        src_vocab_size, 
        tgt_vocab_size,
        max_position_embeddings=512,
        dropout=0.1,
        bias=True,
        device='cuda',
        prior_var=1.0
    ):
        super().__init__()
        
        self.tgt_vocab_size = tgt_vocab_size
        
        # Source and target embeddings
        self.src_embedding = VariationalEmbeddingWithProjection(
            vocab_size=src_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            prior_var=prior_var,
            device=device
        )
        
        self.tgt_embedding = VariationalEmbeddingWithProjection(
            vocab_size=tgt_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            prior_var=prior_var,
            device=device
        )
        
        # Encoder-Decoder stack
        self.encoder_decoder = VariationalTransformerEncoderDecoder(
            num_layer=num_layer,
            d_model=d_model,
            d_ff=d_ff,
            num_head=num_head,
            dropout=dropout,
            bias=bias,
            prior_var=prior_var,
            device=device
        )
        
        # Output projection and softmax
        self.output_projection = VariationalLinearLayer(d_model, tgt_vocab_size, bias=True, prior_var=prior_var, device=device)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def shift_target_right(self, tgt_tokens):
        # Shift target tokens right by padding with zeros at the beginning
        batch_size, seq_len = tgt_tokens.size()
        
        # Create start token (zeros)
        start_tokens = torch.zeros(batch_size, 1, dtype=tgt_tokens.dtype, device=tgt_tokens.device)
        
        # Concatenate start token and remove last token
        shifted_tokens = torch.cat([start_tokens, tgt_tokens[:, :-1]], dim=1)
        
        return shifted_tokens
        
    def forward(self, src_tokens, tgt_tokens, padding_mask=None):
        """
        Args:
            src_tokens: source sequence [batch_size, src_len]
            tgt_tokens: target sequence [batch_size, tgt_len]
            padding_mask: padding mask [batch_size, 1, 1, seq_len]
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size] log probabilities
        """
        # Shift target tokens right for teacher forcing
        shifted_tgt_tokens = self.shift_target_right(tgt_tokens)
        
        # Embed source and target sequences
        src_embedding = self.src_embedding(src_tokens)
        tgt_embedding = self.tgt_embedding(shifted_tgt_tokens)
        
        # Pass through encoder-decoder stack
        decoder_output = self.encoder_decoder(
            embed_encoder_input=src_embedding,
            embed_decoder_input=tgt_embedding,
            padding_mask=padding_mask
        )
        
        # Project to vocabulary size and apply log softmax
        logits = self.output_projection(decoder_output)
        log_probs = self.softmax(logits)
        
        return log_probs

