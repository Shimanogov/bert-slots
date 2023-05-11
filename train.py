from datetime import datetime

import torch
import wandb

from data_collector import Collector, to_torch, FasterCollector
from envs import Push
from model import SlotBert
from policy import RandomPolicy
from slots import FixSLATE


def print_layers(md):
    sm = 0
    for name, param in md.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            sm += int(param.data.flatten().shape[0])
    print()
    print(sm)
    print()
    return sm


config = {
    # SLATE PARAMS
    'num_slots': 6,
    'vocab_size': 128,
    'd_model': 128,
    'image_size': 64,
    'dropout': 0.1,
    'num_iterations': 3,
    'slot_size': 32,
    'mlp_hidden_size': 192,
    'pos_channels': 4,
    'num_slot_heads': 1,
    'num_dec_blocks': 4,
    'num_heads': 4,
    # MODEL PARAMS
    'time': 8,
    'n_bert_heads': 4,
    'dim_feedforward': 256,
    'num_layers': 4,
    'detach': True,
    # TRAINING PARAMS
    'batch_size': 128,
    'num_epochs': 1000 * 100 + 1,
    'inference_every': 100,
    'lr_period': 100,
    'faster': True
}

if __name__ == '__main__':
    wandb.init(config=config, project='SlotBert')
    env = Push(return_state=False)
    policy = RandomPolicy(env)
    if config['faster']:
        collector = FasterCollector(policy)
    else:
        collector = Collector(policy)
    slate = FixSLATE(image_size=config['image_size'], num_slots=config['num_slots'], slot_size=config['slot_size'],
                     vocab_size=config['vocab_size'], d_model=config['d_model'], dropout=config['dropout'],
                     num_iterations=config['num_iterations'], mlp_hidden_size=config['mlp_hidden_size'],
                     pos_channels=config['pos_channels'], num_slot_heads=config['num_slot_heads'],
                     num_dec_blocks=config['num_dec_blocks'], num_heads=config['num_heads'])
    model = SlotBert(slate, env.action_space.n, time=config['time'], n_heads=config['n_bert_heads'],
                     dim_feedforward=config['dim_feedforward'], num_layers=config['num_layers'],
                     detach=config['detach'])
    model.to('cuda')

    # TODO: set up lr scheldue
    optimizer = torch.optim.Adam(model.parameters())

    num_tokens = config['batch_size'] * config['time'] * (config['num_slots'] + 2)
    tokens_passed = 0
    interactions_passed = 0
    total_params = print_layers(model)
    for i in range(config['num_epochs']):
        wandb_log = {}
        start_time = datetime.now()
        obses, actions, rewards = collector.collect_batch(target_len=config['time'],
                                                          batch_size=config['batch_size'])
        collected_time = datetime.now()

        obses, actions, rewards = to_torch(obses, actions, rewards, device='cuda')

        optimizer.zero_grad()
        new_tokens, losses = model(obses, actions, rewards)
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        end_time = datetime.now()
        tokens_passed += num_tokens
        interactions_passed += config['batch_size'] * config['time']

        wandb_log['LOSSES/total'] = loss
        wandb_log['LOSSES/slate_mse'] = losses[0]
        wandb_log['LOSSES/slate_crossentropy'] = losses[1]
        wandb_log['LOSSES/bert_mse'] = losses[2]
        wandb_log['LOSSES/decoder_crossentropy'] = losses[3]
        wandb_log['TIME/collecting_batch'] = (collected_time - start_time).seconds
        wandb_log['TIME/using_model'] = (end_time - collected_time).seconds
        wandb_log['LEARNING/epochs_passes'] = i
        wandb_log['LEARNING/tokens_passes'] = tokens_passed
        wandb_log['LEARNING/interaction_passes'] = interactions_passed
        wandb_log['LEARNING/tokens_to_params'] = tokens_passed / total_params
        wandb_log['LEARNING/percent epochs'] = i / config['num_epochs']

        if i % config['inference_every'] == 0:
            start_time = datetime.now()
            obses, actions, rewards = collector.collect_batch(target_len=config['time'],
                                                              batch_size=config['batch_size'])
            collected_time = datetime.now()
            obses, actions, rewards = to_torch(obses, actions, rewards, device='cuda')
            inv_results = model.inv_din_inference(obses, actions, rewards)
            forw_results = model.forw_din_inference(obses, actions, rewards)
            end_time = datetime.now()

            for k, v in inv_results.items():
                wandb_log['INV_DIN/'+k] = v
            for k, v in forw_results.items():
                wandb_log['FORW_DIN/'+k] = v

            wandb_log['TIME/collecting_inference_batch'] = (collected_time - start_time).seconds
            wandb_log['TIME/inference_model'] = (end_time - collected_time).seconds

        wandb.log(wandb_log)
