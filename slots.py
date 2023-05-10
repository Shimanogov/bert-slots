from argparse import Namespace

import torch
from slate.slate import SLATE
import torch.nn.functional as F

from slate.utils import gumbel_softmax

class FixSLATE(SLATE):
    def __init__(self, image_size, num_slots, slot_size,
                 vocab_size=512, d_model=128, dropout=0.1, num_iterations=3, mlp_hidden_size=256, pos_channels=8,
                 num_slot_heads=1, num_dec_blocks=2, num_heads=4):
        args = Namespace()
        args.num_slots = num_slots
        args.vocab_size = vocab_size
        args.d_model = d_model
        args.image_size = image_size
        args.dropout = dropout
        args.num_iterations = num_iterations
        args.slot_size = slot_size
        args.mlp_hidden_size = mlp_hidden_size
        args.pos_channels = pos_channels
        args.num_slot_heads = num_slot_heads
        args.num_dec_blocks = num_dec_blocks
        args.num_heads = num_heads

        self.slot_size = slot_size
        self.num_slots = num_slots
        super().__init__(args)

    def forward(self, image, tau=0.15, hard=True):
        """
        image: batch_size x img_channels x H x W
        """

        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)
        _, _, H_enc, W_enc = z_logits.size()
        z = gumbel_softmax(z_logits, tau, hard, dim=1)

        # dvae recon
        recon = self.dvae.decoder(z)
        mse = torch.mean(((image - recon) ** 2))

        # hard z
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()

        # target tokens for transformer
        z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # add BOS token
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input],
                                        dim=-2)
        z_transformer_input[:, 0, 0] = 1.0

        # tokens to embeddings
        emb_input = self.dictionary(z_transformer_input)
        emb_input = self.positional_encoder(emb_input)

        # apply slot attention
        slots, attns = self.slot_attn(emb_input[:, 1:])
        attns = attns.transpose(-1, -2)
        attns = attns.reshape(B, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc,
                                                                                    dim=-2).repeat_interleave(
            W // W_enc, dim=-1)
        attns = image.unsqueeze(1) * attns + 1. - attns

        # apply transformer
        slots_proj = self.slot_proj(slots)
        decoder_output = self.tf_dec(emb_input[:, :-1], slots_proj)
        pred = self.out(decoder_output)
        # cross_entropy = -(z_transformer_target * torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()
        # TODO: check new mean is ok
        cross_entropy = torch.mean(-(z_transformer_target * torch.log_softmax(pred, dim=-1)))
        return (
            recon.clamp(0., 1.),
            cross_entropy,
            mse,
            attns,
            slots
        )
