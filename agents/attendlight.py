from cmath import phase
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent import SharedAgent, IndependentAgent
from agents.pfrl_dqn import DQNAgent
from signal_config import signal_configs
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.agents import REINFORCE

from agents.agent import IndependentAgent, Agent


# class AttendLight(IndependentAgent):
#     def __init__(self, config, obs_act, map_name, thread_number):
#         super().__init__(config, obs_act, map_name, thread_number)
#         for key in obs_act:
#             phase_pairs = signal_configs[map_name]['phase_pairs']
#             num_actions = obs_act[key][1]

#             valid_acts = signal_configs[map_name]['valid_acts']
#             if valid_acts is None:
#                 valid_acts = dict((i, i) for i in range(len(phase_pairs)))
#             else:
#                 valid_acts = valid_acts[key]
#                 valid_phase_number = list(valid_acts.keys())
#                 phase_pairs = [phase_pairs[i] for i in valid_phase_number]

#             model = AttendModel_independent(config, valid_acts, phase_pairs, self.device)

#             self.agents[key] = DQNAgent(config, num_actions, model)


class REINFORCEagent(Agent):
    def __init__(self, config, num_actions, model):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        self.agent = REINFORCE(self.model, self.optimizer, gpu=self.device.index,
                         max_grad_norm=1.0)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')

class AttendLight(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        phase_pairs = signal_configs[map_name]['phase_pairs']
        num_actions = len(phase_pairs)

        self.valid_acts = signal_configs[map_name]['valid_acts']

        model = AttendModel_shared(config, num_actions, phase_pairs, self.device)
        self.agent = DQNAgent(config, num_actions, model, num_agents=config['num_lights'])


class AttendModel_shared(nn.Module):
    def __init__(self, config, output_shape, phase_pairs, device):
        super(AttendModel_shared, self).__init__()
        self.phase_pairs = phase_pairs
        self.oshape = output_shape

        self.device = device
        self.demand_shape = config['demand_shape']      # Allows more than just queue to be used

        self.lane_embed_units = 128

        self.lane_embedding = nn.Linear(self.demand_shape, self.lane_embed_units)

        self.attn = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        self.state_attn_query = nn.Linear(self.lane_embed_units, self.lane_embed_units)
        self.state_attn_refer = nn.Linear(self.lane_embed_units, self.lane_embed_units)
        self.state_attn = nn.Linear(self.lane_embed_units, 1)

        self.action_attn_query = nn.Linear(self.lane_embed_units, self.lane_embed_units)
        self.action_attn_refer = nn.Linear(self.lane_embed_units, self.lane_embed_units)    # RNN
        self.action_attn = nn.Linear(self.lane_embed_units, 1)

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        states = states.to(self.device)
        n_lanes = int((states.size()[1]-1)/self.demand_shape)

        phases = states[:, 0]
        states = states[:, 1:]
        states = states.float()

        # Embedding
        lane_embeds = []
        for i in range(n_lanes):
            lane_idx = i * self.demand_shape
            l_embed = self.lane_embedding(states[:, lane_idx:lane_idx+self.demand_shape])
            lane_embeds.append(l_embed)
        lane_embeds = torch.stack(lane_embeds, 1)
        
        # State Attention
        lane_reference = torch.mean(lane_embeds, dim=(1, ))
        lane_reference = self.state_attn_refer(lane_reference)
        lane_reference = lane_reference.unsqueeze(dim=1).tile(n_lanes, 1)
        lane_query = self.state_attn_query(lane_embeds)
        lane_attn = self.state_attn(self.attn(lane_reference + lane_query))

        lane_attn = self.softmax(lane_attn).tile(self.lane_embed_units)

        lane_results = lane_attn * lane_embeds

        # Create Phase vectors
        pairs = []
        for pair in self.phase_pairs:
            pairs.append(lane_results[:, pair[0]] + lane_results[:, pair[1]])
        pairs = torch.stack(pairs, 1)

        # Create reference for action attention
        green_lights = []
        for i, p in enumerate(phases):
            p = int(p.item())
            green_lights.append(pairs[i, p])
        green_lights = torch.stack(green_lights, 1).permute((1, 0))
        green_lights = self.action_attn_refer(green_lights)
        green_lights = green_lights.unsqueeze(dim=1).tile(len(self.phase_pairs), 1)

        pairs = self.action_attn_query(pairs)

        action_attn = self.action_attn(self.attn(pairs + green_lights))
        action_attn = torch.squeeze(action_attn, dim=-1)

        return self.head(action_attn)

class AttendModel_independent(nn.Module):
    def __init__(self, config, valid_acts, phase_pairs, device):
        super(AttendModel_independent, self).__init__()
        self.phase_pairs = phase_pairs
        self.valid_acts = valid_acts
        if self.valid_acts is not None:
            self.valid_acts_swapped = dict((v, k) for k, v in self.valid_acts.items())

        self.device = device
        self.demand_shape = config['demand_shape']      # Allows more than just queue to be used

        self.valid_lanes = list(set(np.reshape(self.phase_pairs, (-1))))

        self.lane_dict = dict()
        for lane in self.valid_lanes:
            self.lane_dict[lane] = len(self.lane_dict)

        self.lane_embed_units = 128

        self.lane_embedding = nn.Linear(self.demand_shape, self.lane_embed_units)

        self.attn = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        self.state_attn_query = nn.Linear(self.lane_embed_units, self.lane_embed_units)
        self.state_attn_refer = nn.Linear(self.lane_embed_units, self.lane_embed_units)
        self.state_attn = nn.Linear(self.lane_embed_units, 1)

        self.action_attn_query = nn.Linear(self.lane_embed_units, self.lane_embed_units)
        self.action_attn_refer = nn.Linear(self.lane_embed_units, self.lane_embed_units)    # RNN
        self.action_attn = nn.Linear(self.lane_embed_units, 1)

        self.head = DiscreteActionValueHead()

    def forward(self, states):
        states = states.to(self.device)
        n_lanes = int((states.size()[1]-1)/self.demand_shape)

        if self.valid_acts is not None:
            phases = []
            for p in states[:, 0]:
                phase = self.valid_acts_swapped[int(p.item())]
                phases.append(phase)
            phases = torch.tensor(phases)
        else:
            phases = states[:, 0]
        states = states[:, 1:]
        states = states.float()

        # Embedding
        lane_embeds = []
        for i in self.valid_lanes:
            lane_idx = i * self.demand_shape
            l_embed = self.lane_embedding(states[:, lane_idx:lane_idx+self.demand_shape])
            lane_embeds.append(l_embed)
        lane_embeds = torch.stack(lane_embeds, 1)
        
        # State Attention
        lane_reference = torch.mean(lane_embeds, dim=(1, ))
        lane_reference = self.state_attn_refer(lane_reference)
        lane_reference = lane_reference.unsqueeze(dim=1).tile(len(self.valid_lanes), 1)
        lane_query = self.state_attn_query(lane_embeds)
        lane_attn = self.state_attn(self.attn(lane_reference + lane_query))

        lane_attn = self.softmax(lane_attn).tile(self.lane_embed_units)

        lane_results = lane_attn * lane_embeds

        # Create Phase vectors
        pairs = []
        for pair in self.phase_pairs:
            pairs.append(lane_results[:, self.lane_dict[pair[0]]] + lane_results[:, self.lane_dict[pair[1]]])
        pairs = torch.stack(pairs, 1)

        # Create reference for action attention
        green_lights = []
        for i, p in enumerate(phases):
            p = int(p.item())
            if self.valid_acts is not None:
                p = self.valid_acts[p]
            green_lights.append(pairs[i, p])
        green_lights = torch.stack(green_lights, 1).permute((1, 0))
        green_lights = self.action_attn_refer(green_lights)
        green_lights = green_lights.unsqueeze(dim=1).tile(len(self.phase_pairs), 1)

        pairs = self.action_attn_query(pairs)

        action_attn = self.action_attn(self.attn(pairs + green_lights))
        action_attn = torch.squeeze(action_attn, dim=-1)

        return self.head(action_attn)