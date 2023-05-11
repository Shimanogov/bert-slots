import torch
import wandb
from torch import nn
from dataclasses import dataclass
import torchvision.utils as vutils


def get_time_emb(dim, time):
    pos = torch.arange(0, time, dtype=torch.float)
    omega = torch.arange(dim // 2, dtype=torch.float)
    omega /= dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concatenate([emb_sin, emb_cos], dim=1)
    return emb


class SlotBert(nn.Module):
    def __init__(self, slate, num_actions, time,
                 n_heads=4, dim_feedforward=512, num_layers=4, detach=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detach = detach
        self.slate = slate
        self.action_emb = nn.Embedding(num_actions, slate.slot_size)
        self.rew_emb = nn.Linear(1, slate.slot_size)
        self.modality_mask_emb = nn.Embedding(4, slate.slot_size)
        self.time_emb = get_time_emb(slate.slot_size, time)
        bert_layer = nn.TransformerEncoderLayer(slate.slot_size, n_heads, dim_feedforward, batch_first=True)
        self.bert = nn.TransformerEncoder(bert_layer, num_layers)
        self.act_ff = nn.Sequential(nn.Linear(slate.slot_size, slate.slot_size),
                                    nn.GELU(),
                                    nn.Linear(slate.slot_size, num_actions),
                                    )
        self.act_loss = torch.nn.CrossEntropyLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_sep(self, obses, actions, rew):
        target_sizes = list(obses.shape)[:2]
        obses = torch.flatten(obses, 0, 1)
        recon, ce, mse, attns, obses = self.slate(obses)
        if self.detach:
            obses = obses.detach()
        obses_ret = torch.unflatten(obses, 0, target_sizes)
        return obses_ret, self.action_emb(actions), self.rew_emb(rew.unsqueeze(-1)), mse, ce

    def mask_sep(self, obses, actions, rew):
        # 0 prob - 0 out
        # 0 - mask (all masked)
        gamma = torch.rand((obses.shape[0]), device=self.device)
        gamma = gamma.unsqueeze(-1)
        mask_probs_obses = torch.ones(obses.shape[:-1], device=self.device) * gamma.unsqueeze(-1)
        mask_probs_actions_rew = torch.ones(actions.shape[:-1], device=self.device) * gamma
        mask_obses = torch.bernoulli(mask_probs_obses).long()
        mask_actions = torch.bernoulli(mask_probs_actions_rew).long()
        mask_rew = torch.bernoulli(mask_probs_actions_rew).long()

        return (
                   (obses * mask_obses.unsqueeze(-1)) + (1 - mask_obses.unsqueeze(-1)) * self.modality_mask_emb(
                       mask_obses),
                   (actions * mask_actions.unsqueeze(-1)) + (1 - mask_actions.unsqueeze(-1)) * self.modality_mask_emb(
                       mask_actions),
                   (rew * mask_rew.unsqueeze(-1)) + (1 - mask_rew.unsqueeze(-1)) * self.modality_mask_emb(mask_rew),
               ), (mask_obses, mask_actions, mask_rew)

    def add_modality_sep(self, obses, actions, rew):
        mod_obses = self.modality_mask_emb(torch.ones(obses.shape[:-1], dtype=torch.long, device=self.device))
        mod_actions = self.modality_mask_emb(torch.ones(actions.shape[:-1], dtype=torch.long, device=self.device) * 2)
        mod_rew = self.modality_mask_emb(torch.ones(rew.shape[:-1], dtype=torch.long, device=self.device) * 3)
        return obses + mod_obses, actions + mod_actions, rew + mod_rew

    def add_time_sep(self, obses, actions, rew):
        actions_rew_time = self.time_emb.unsqueeze(0).to(self.device)
        obses_time = actions_rew_time.unsqueeze(-2)
        return obses + obses_time, actions + actions_rew_time, rew + actions_rew_time

    def concat_all(self, obses, actions, rew):
        actions_new = actions.unsqueeze(2)
        rew_new = rew.unsqueeze(2)
        stack = torch.cat([obses, actions_new, rew_new], dim=2)
        stack = torch.flatten(stack, start_dim=1, end_dim=2)
        return stack

    def sep_to_seq(self, obses, actions, rewards):
        obses, actions, rewards = self.add_modality_sep(obses, actions, rewards)
        obses, actions, rewards = self.add_time_sep(obses, actions, rewards)
        return self.concat_all(obses, actions, rewards)

    def pass_to_bert(self, seq):
        return self.bert(seq)

    def forward(self, obses, actions, rewards):
        t_obses, t_actions, t_rewards, mse, ce = self.embed_sep(obses, actions, rewards)
        (m_obses, m_actions, m_rewards), (bm_o, bm_a, bm_r) = self.mask_sep(t_obses, t_actions, t_rewards)
        # tokens = self.sep_to_seq(t_obses, t_actions, t_rewards)
        masked_tokens = self.sep_to_seq(m_obses, m_actions, m_rewards)
        masks = 1 - self.concat_all(bm_o, bm_a, bm_r)  # mask = 0 should be included in loss
        new_tokens = self.pass_to_bert(masked_tokens)
        bert_mse = torch.mean((new_tokens - masked_tokens) ** 2 * masks.unsqueeze(-1))
        if self.detach:
            new_tokens = new_tokens.detach()

        # TODO: check loss is correct
        new_ttokens = new_tokens[:, self.slate.num_slots::self.slate.num_slots + 2]
        actions_time = self.time_emb.unsqueeze(0).to(self.device)
        mod_actions = self.modality_mask_emb(torch.ones(new_ttokens.shape[:-1],
                                                        dtype=torch.long, device=self.device) * 2)
        new_ttokens = new_ttokens - actions_time - mod_actions
        new_actions = self.act_ff(new_ttokens)
        loss = self.act_loss(new_actions.flatten(0, 1), actions.flatten(0, 1))
        # END OF TD

        return new_tokens, (mse, ce, bert_mse, loss)

    @torch.no_grad()
    def inv_din_inference(self, obses, actions, rewards):
        losses = {}
        # we should mask all rew
        # we should mask last 2 actions
        # we should not mask obses
        # meaningful actions: last 2 obses are different
        meaningful = torch.abs(obses[:, -1] - obses[:, -2])
        meaningful = torch.max(torch.flatten(meaningful, 1), 1).values
        meaningful = torch.eq(meaningful, torch.zeros_like(meaningful))

        t_obses, t_actions, t_rewards, _, _ = self.embed_sep(obses, actions, rewards)
        mask_obses = torch.ones(t_obses.shape[:-1], device=self.device).long()
        mask_rew = torch.zeros(t_rewards.shape[:-1], device=self.device).long()
        mask_actions = torch.ones(t_actions.shape[:-1], device=self.device).long()
        mask_actions[:, -1] = 0
        mask_actions[:, -2] = 0
        m_obses, m_actions, m_rewards = (
            (t_obses * mask_obses.unsqueeze(-1)) + (1 - mask_obses.unsqueeze(-1)) * self.modality_mask_emb(
                mask_obses),
            (t_actions * mask_actions.unsqueeze(-1)) + (1 - mask_actions.unsqueeze(-1)) * self.modality_mask_emb(
                mask_actions),
            (t_rewards * mask_rew.unsqueeze(-1)) + (1 - mask_rew.unsqueeze(-1)) * self.modality_mask_emb(mask_rew),
        )
        masked_tokens = self.sep_to_seq(m_obses, m_actions, m_rewards)
        new_tokens = self.pass_to_bert(masked_tokens)
        new_ttokens = new_tokens[:, self.slate.num_slots::self.slate.num_slots + 2]
        actions_time = self.time_emb.unsqueeze(0).to(self.device)
        mod_actions = self.modality_mask_emb(torch.ones(new_ttokens.shape[:-1],
                                                        dtype=torch.long, device=self.device) * 2)
        new_ttokens = new_ttokens - actions_time - mod_actions
        old_ttokens = masked_tokens[:, self.slate.num_slots::self.slate.num_slots + 2]
        old_ttokens = old_ttokens - actions_time - mod_actions
        new_actions = self.act_ff(new_ttokens)

        new_action_emb = new_ttokens[:, -2]
        old_action_emb = old_ttokens[:, -2]
        losses['mse'] = torch.mean((new_action_emb - old_action_emb) ** 2)
        losses['meaningful mse'] = torch.mean(((new_action_emb - old_action_emb)[meaningful]) ** 2)

        distance = torch.norm(new_action_emb.unsqueeze(1) - self.action_emb.weight.data.unsqueeze(0), dim=2)
        nearest = torch.argmin(distance, dim=1)

        new_action = new_actions[:, -2]
        old_action = actions[:, -2]
        losses['cross entropy'] = self.act_loss(new_action, old_action)
        losses['meanigful cross entropy'] = self.act_loss(new_action[meaningful],
                                                          old_action[meaningful])

        new_action_max = torch.max(new_action, dim=1).indices
        losses['accuracy'] = torch.sum(torch.eq(old_action, new_action_max)) / (old_action.shape[0])
        losses['meanigful accuracy'] = torch.sum(torch.eq(old_action[meaningful],
                                                          new_action_max[meaningful])) / (
                                       old_action[meaningful].shape[0])
        losses['nearest accuracy'] = torch.sum(torch.eq(old_action, nearest)) / (old_action.shape[0])
        losses['nearest meanigful accuracy'] = torch.sum(torch.eq(old_action[meaningful],
                                                                  nearest[meaningful])) / (
                                                   old_action[meaningful].shape[0])

        return losses

    @torch.no_grad()
    def forw_din_inference(self, obses, actions, rewards):
        # we should mask all rew
        # we should not mask actions
        # we should mask last obs
        losses = {}
        t_obses, t_actions, t_rewards, _, _ = self.embed_sep(obses, actions, rewards)
        mask_obses = torch.ones(t_obses.shape[:-1], device=self.device).long()
        mask_rew = torch.zeros(t_rewards.shape[:-1], device=self.device).long()
        mask_actions = torch.ones(t_actions.shape[:-1], device=self.device).long()
        mask_obses[:, -1] = 0

        m_obses, m_actions, m_rewards = (
            (t_obses * mask_obses.unsqueeze(-1)) + (1 - mask_obses.unsqueeze(-1)) * self.modality_mask_emb(
                mask_obses),
            (t_actions * mask_actions.unsqueeze(-1)) + (1 - mask_actions.unsqueeze(-1)) * self.modality_mask_emb(
                mask_actions),
            (t_rewards * mask_rew.unsqueeze(-1)) + (1 - mask_rew.unsqueeze(-1)) * self.modality_mask_emb(mask_rew),
        )
        masked_tokens = self.sep_to_seq(m_obses, m_actions, m_rewards)
        new_tokens = self.pass_to_bert(masked_tokens)
        new_tokens = new_tokens.unflatten(1, (-1, self.slate.num_slots + 2))
        old_tokens = masked_tokens.unflatten(1, (-1, self.slate.num_slots + 2))

        new_slots = new_tokens[:, -1, :-2]
        old_slots = old_tokens[:, -1, :-2]
        losses['mse slots'] = torch.mean((new_slots - old_slots) ** 2)
        new_slots_deemb = new_slots - self.modality_mask_emb(torch.ones(new_slots.shape[:-1],
                                                                        dtype=torch.long,
                                                                        device=self.device))
        new_slots_deemb = new_slots_deemb - self.time_emb.to(self.device).unsqueeze(-2)[-1:]
        old_slots_deemb = old_slots - self.modality_mask_emb(torch.ones(old_slots.shape[:-1],
                                                                        dtype=torch.long,
                                                                        device=self.device))
        old_slots_deemb = old_slots_deemb - self.time_emb.to(self.device).unsqueeze(-2)[-1:]
        reconstruct = self.slate.reconstruct_slots(new_slots_deemb)
        reconstruct_old = self.slate.reconstruct_slots(old_slots_deemb)
        losses['mse images'] = torch.mean((reconstruct - reconstruct_old) ** 2)
        reconstruct = torch.cat([reconstruct[:32], reconstruct_old[:32]], dim=0)
        grid = vutils.make_grid(reconstruct, nrow=2, pad_value=0.2)[:, 2:-2, 2:-2]
        losses['visualisation'] = wandb.Image(grid)
        # TODO: add logging of ground truth

        return losses
