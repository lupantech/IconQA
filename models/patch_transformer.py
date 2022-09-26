"""
The visual transformer code is built on top of the following paper and project:
- Paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- Project: https://github.com/gupta-abhay/ViT
"""
import torch
import torch.nn as nn
from .vitrm_models.Transformer import TransformerModel
from .vitrm_models.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from datetime import datetime
class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_patches,
        patch_dim,
        out_dim,
        return_all_embeddings,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_representation=True,
        positional_encoding_type="learned",
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.return_all_embeddings = return_all_embeddings
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.seq_length = num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim)) # convert the Tensor a module parameter

        self.linear_encoding = nn.Linear(patch_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, out_dim)

        self.to_cls_token = nn.Identity()

    def forward(self, x):
        # x: [batch, patch_nums, patch_dim] ([N,patch_nums,2048])
        x = self.linear_encoding(x) # [N, patch_nums, 768]

        # position encoding
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # [N,patch_nums,768]
        x = torch.cat((cls_tokens, x), dim=1) # [N,patch_nums+1,768]
        x = self.position_encoding(x) # [N,patch_nums+1,768]
        x = self.pe_dropout(x)

        # apply transformer
        x = self.transformer(x) # [N,patch_nums+1,768]

        return x


def Patch_Transformer(num_patches=30, num_heads=4, num_layers=1, patch_emb_dim=768):

    return VisionTransformer(
        num_patches=num_patches,
        patch_dim=2048,
        out_dim=2048,
        return_all_embeddings=True,
        embedding_dim=patch_emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=2048,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=True,
        positional_encoding_type="learned",
    )


if __name__ == '__main__':
    model = Patch_Transformer()
    input = torch.rand(4, 30, 2048) # [batch, patch_nums, patch_dim]
    output = model(input)
    print(datetime.now().isoformat(), " ",output.size())
