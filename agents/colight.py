
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent import IndependentAgent, Agent
from agents.pfrl_dqn import DQNAgent
from signal_config import signal_configs
from pfrl.q_functions import DiscreteActionValueHead

from agents.pfrl_ppo import PFRLPPOAgent
from pfrl.policies import SoftmaxCategoricalHead

GEO = False

if GEO:
    from signal_configs_geo import signal_configs_geo

class CoLight(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        self.config = config
        self.agent = None
        self.valid_acts = None
        self.reverse_valid = None

        phase_pairs = signal_configs[map_name]['phase_pairs']
        num_actions = len(phase_pairs)

        if GEO:
            signals = signal_configs_geo[map_name]
        else:
            signals = signal_configs[map_name]

        n_signal = len(list(signals.keys())[2:])

        if not GEO:
            tmp = dict()
            for s in list(signals.keys())[2:] :
                tmp[s] = len(tmp)
            self.adj = np.zeros((n_signal, 5, n_signal))
            for s in list(signals.keys())[2:] :
                for i, d in enumerate(signals[s]['downstream'].values()) :
                    if d is not None:
                        self.adj[tmp[s], i+1, tmp[d]] = 1
                for i in range(n_signal) :
                    self.adj[i, 0, i] = 1
        else:
            tmp = dict()
            for s in list(signals.keys())[:-2] :
                tmp[s] = len(tmp)
            self.adj = np.zeros((n_signal, 1, n_signal))
            for s in list(signals.keys())[:-2] :
                for d in signals[s]['geo_neighbor'] :
                    if d is not None:
                        self.adj[tmp[s], 0, tmp[d]] = 1
                for i in range(n_signal) :
                    self.adj[i, 0, i] = 1


        self.obs_size = obs_act[list(obs_act.keys())[0]][0][1]

        
        self.head = int(self.config['N_HEAD'])
        self.latent_dim = int(self.config['N_DIM'])
        print(self.head, self.latent_dim)
        self.total_dim = self.head * self.latent_dim

        self.mhgat = [
                CoLightMultiHeadGAT(self.head, self.latent_dim).to(self.device),  
                # nn.Linear(self.latent_dim, self.total_dim),
                # CoLightMultiHeadGAT(self.head, self.latent_dim),
                # nn.Linear(self.latent_dim, self.total_dim),
                # CoLightMultiHeadGAT(self.head, self.latent_dim),
                # nn.Linear(self.latent_dim, self.total_dim),
            ]
        
        # self.valid_acts = signal_configs[map_name]['valid_acts']
        # model = CoLightAgent(config, self.obs_size, num_actions, self.device, self.adj)
        # self.agent = DQNAgent(config, num_actions, model, num_agents=config['num_lights'])

        for key in obs_act:
            act_space = obs_act[key][1]
            
            model = CoLightAgent(config, self.obs_size, act_space, self.device, self.adj, gats=self.mhgat).to(self.device)

            if bool(self.config['use_ppo']) == True:
                self.agents[key] = PFRLPPOAgent(config, act_space, model)
            else:
                self.agents[key] = DQNAgent(config, act_space, model)

    def observe(self, observation, reward, done, info):
        # if done:
        #     if info['eps'] % 10 == 0:
        #         for i in range(len(self.mhgat)):

        for agent_id in observation.keys():
            self.agents[agent_id].observe(observation[agent_id], reward[agent_id], done, info)
            if done:
                if info['eps'] == 1 or info['eps'] % 5 == 0:
                    for i in range(len(self.mhgat)):
                        gat_path = self.config['log_dir']+'agent_' + str(info['eps']) + '_gat' + str(i) + '_' + agent_id + '.pt'
                        print("Save GAT at " + gat_path)
                        torch.save(self.mhgat[i].state_dict(), gat_path)

                    model_path = self.config['log_dir']+'agent_' + str(info['eps']) + '_'  + agent_id
                    print("Save model at " + model_path)
                    self.agents[agent_id].save(model_path)


class CoLightAgent(nn.Module):
    def __init__(self, config, obs_space, act_space, device, adj, gats=None):
        super(CoLightAgent, self).__init__()

        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.adj = torch.tensor(adj, dtype=torch.float).to(self.device)

        self.head = int(self.config['N_HEAD'])
        self.latent_dim = int(self.config['N_DIM'])
        self.total_dim = self.head * self.latent_dim

        if gats is None:
            raise Error
        else:
            self.mhgat = gats

        self.obs_embed = nn.Sequential(
            nn.Linear(self.obs_space, self.total_dim),
            nn.ReLU(),
            nn.Linear(self.total_dim, self.total_dim),
            nn.ReLU(),
        )

        self.action_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.act_space),
        )

        self.softmax = nn.Softmax(dim=-1)

        if bool(self.config['use_ppo']) == True:
            self.act_head = SoftmaxCategoricalHead()
            self.critic_layer = nn.Sequential(
                nn.Linear(self.latent_dim, 1),
            )
        else:
            self.act_head = DiscreteActionValueHead()

    def forward(self, x):
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(0)
        target_index = x[:, -1, 0]

        obs = x[:, :-1, :]

        ### Embedding ###
        out = self.obs_embed(obs)

        ### Attention ###
        for n, layer in enumerate(self.mhgat):
            if n % 2 == 0:
                out = layer(out, self.adj)
            else:
                out = layer(out)

        ### Action Selection ###
        target_indices = [[i for _ in  range(self.latent_dim)] for i in target_index]
        target_indices = torch.tensor(target_indices, dtype=torch.int64).unsqueeze(axis=-1).reshape(len(target_index), 1, self.latent_dim).to(self.device)
        gather_out = torch.gather(out, 1, target_indices).squeeze(1)

        act = self.action_layer(gather_out)
        act = self.act_head(act)

        if bool(self.config['use_ppo']) == True:
            cri = self.critic_layer(gather_out)
        
            return act, cri
        else:
            return act

class CoLightMultiHeadGAT(nn.Module):
    def __init__(self, n_heads, n_dim):
        super(CoLightMultiHeadGAT, self).__init__()

        self.head = n_heads
        self.latent_dim = n_dim
        self.total_dim = n_heads * n_dim
        
        self.local_attn_heads = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.ReLU()
        ) 
        
        self.neighbor_attn_heads = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.ReLU()
        )

        self.neighbor_hidden_layer = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedded, adj_matrix):
        # (batch_size, n_intersection, dim) -> (batch_size, n_intersection, dim)

        batch_size = embedded.shape[0]
        n_intersection = embedded.shape[1]
        
        neighbor_embed = torch.tile(embedded.unsqueeze(1), (1, n_intersection, 1, 1))
        neighbor_embed = torch.matmul(adj_matrix, neighbor_embed)

        agent_heads = self.local_attn_heads(embedded)
        agent_heads = torch.reshape(agent_heads, (batch_size, n_intersection, 1, self.latent_dim, self.head))
        agent_heads = agent_heads.permute((0, 1, 4, 2, 3))

        neighbor_heads = self.neighbor_attn_heads(neighbor_embed)
        neighbor_heads = torch.reshape(neighbor_heads, (batch_size, n_intersection, -1, self.latent_dim, self.head))
        neighbor_heads = neighbor_heads.permute((0, 1, 4, 2, 3))

        attn = torch.einsum("ijklm,ijknm->ijklm", agent_heads, neighbor_heads)
        attn = self.softmax(attn)

        neighbor_hidden = self.neighbor_hidden_layer(neighbor_embed)
        neighbor_hidden = torch.reshape(neighbor_hidden, (batch_size, n_intersection, -1, self.latent_dim, self.head))
        neighbor_hidden = neighbor_hidden.permute((0, 1, 4, 2, 3))

        out = torch.einsum("ijklm,ijknm->ijklm", attn, neighbor_hidden)
        out = torch.mean(out, dim=2)
        out = torch.reshape(out, (batch_size, n_intersection, -1))
        return out