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

from agents.module.egt import EGT_Layer

class CoLight_edge(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        self.config = config
        self.agent = None
        self.valid_acts = None
        self.reverse_valid = None

        phase_pairs = signal_configs[map_name]['phase_pairs']
        num_actions = len(phase_pairs)

        signals = signal_configs[map_name]

        n_signal = len(list(signals.keys())[2:])

        # adjacency matrix


        # 방향에 따라 Adjacency 적용

        # tmp = dict()
        # for s in list(signals.keys())[2:] :
        #     tmp[s] = len(tmp)
        # self.adj = np.zeros((n_signal, 5, n_signal))
        # for s in list(signals.keys())[2:] :
        #     for i, d in enumerate(signals[s]['downstream'].values()) :
        #         if d is not None:
        #             self.adj[tmp[s], i+1, tmp[d]] = 1
        #     for i in range(n_signal) :
        #         self.adj[i, 0, i] = 1

        USE_VNODE = bool(self.config['use_vnode'])

        tmp = dict()
        sorted_signals = sorted(list(signals.keys())[2:])
        if not USE_VNODE:   
        # 일반적인 형태
            for s in  sorted_signals:
                tmp[s] = len(tmp)
            self.adj = np.zeros((n_signal, n_signal))
            for s in sorted_signals :
                for i, d in enumerate(signals[s]['downstream'].values()):
                    if d is not None:
                        self.adj[tmp[s], tmp[d]] = 1
        else:
        # None 을 Vnode로 처리
            tmp = dict()
            v_idx = n_signal
            for s in  sorted_signals:
                tmp[s] = len(tmp)
                for d in ['S', 'W', "N", "E"]:
                    if signals[s]['downstream'][d] is None:
                        n_signal += 1
            self.adj = np.zeros((n_signal, n_signal))
            for s in sorted_signals :
                for i, d in enumerate(signals[s]['downstream'].values()):
                    if d is not None:
                        self.adj[tmp[s], tmp[d]] = 1
                    else:
                        self.adj[tmp[s], v_idx] = 1
                        v_idx += 1

        # SVD Postional Encoding
        u, s, vh = np.linalg.svd(self.adj)
        r = 6

        if r < n_signal:
            s = s[:r]
            u = u[:,:r]
            vh = vh[:r,:]
            
            self.position_encodings = np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1)
        elif r > n_signal:
            z = np.zeros((n_signal,r-n_signal,2),dtype=np.float32)
            self.position_encodings = np.concatenate((np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1), z), axis=1)
        else:
            self.position_encodings = np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1)

        #### SVD end ###

        self.obs_size = obs_act[list(obs_act.keys())[0]][0][1]
        
        self.head = int(self.config['N_HEAD'])
        self.latent_dim = int(self.config['N_DIM'])
        # print(self.head, self.latent_dim)
        # self.total_dim = self.head * self.latent_dim

        self.adj_embedding = nn.Sequential(
            nn.Linear(r*2, self.latent_dim)
        ).to(self.device)

        self.node_obs_embed = nn.Sequential(
            nn.Linear(13, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        ).to(self.device)

        self.edge_obs_embed = nn.Sequential(
            nn.Linear(4, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        ).to(self.device)

        self.mhgat = [
                EGT_Layer(self.latent_dim, self.latent_dim, self.head, return_attn_weight=True).to(self.device),
            ]
        
        # self.valid_acts = signal_configs[map_name]['valid_acts']
        # model = CoLightAgent(config, self.obs_size, num_actions, self.device, self.adj)
        # self.agent = DQNAgent(config, num_actions, model, num_agents=config['num_lights'])

        for key in obs_act:
            act_space = obs_act[key][1]
            
            model = CoLightEGTAgent(config = config,
                                    obs_space = self.obs_size, 
                                    act_space = act_space, 
                                    device = self.device, 
                                    pe = self.position_encodings, 
                                    pe_embed = self.adj_embedding, 
                                    node_embed = self.node_obs_embed,
                                    edge_embed = self.edge_obs_embed,
                                    gats =self.mhgat).to(self.device)

            if bool(self.config['use_ppo']) == True:
                self.agents[key] = PFRLPPOAgent(config, act_space, model)
            else:
                self.agents[key] = DQNAgent(config, act_space, model)

    def check_attn_weight(self, x):
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(0)
        with torch.no_grad():
            x = torch.tensor(np.expand_dims(x[list(x.keys())[0]], 0), dtype=torch.float).to(self.device)
            position_encodings_tmp = torch.tensor(self.position_encodings, dtype=torch.float).to(self.device)

            node_size = int(x[0, -1, 0, 0])
            edge_size = int(x[0, -1, 0, 1])
            target_index = x[:, -1, 0, 2]

            node_obs = x[:, 0, :, :node_size]
            edge_obs = x[:, 1:-1, :, :edge_size]

            ### Embedding ###
            node_out = self.node_obs_embed(node_obs)
            pe = self.adj_embedding(position_encodings_tmp.reshape(position_encodings_tmp.shape[0], -1).unsqueeze(0).repeat(x.shape[0], 1, 1))
            node_out = node_out + pe

            edge_out = self.edge_obs_embed(edge_obs)

            ### Attention ###
            for n, layer in enumerate(self.mhgat):
                node_out, edge_out, attn_weight = layer(node_out, edge_out, None)

            return attn_weight


    def get_adj_matrix(self):
        return self.adj

class CoLightEGTAgent(nn.Module):
    def __init__(self, 
                config, 
                obs_space, 
                act_space, 
                device, 
                pe, 
                pe_embed, 
                node_embed,
                edge_embed,
                gats=None):
        super(CoLightEGTAgent, self).__init__()

        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device

        self.position_encodings = torch.tensor(pe, dtype=torch.float).to(device)

        self.head = int(self.config['N_HEAD'])
        self.latent_dim = int(self.config['N_DIM'])
        self.total_dim = self.head * self.latent_dim

        self.mhgat = gats    
        self.pe_embed = pe_embed
        self.node_obs_embed = node_embed
        self.edge_obs_embed = edge_embed

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
        node_size = int(x[0, -1, 0, 0])
        edge_size = int(x[0, -1, 0, 1])
        target_index = x[:, -1, 0, 2]

        node_obs = x[:, 0, :, :node_size]
        edge_obs = x[:, 1:-1, :, :edge_size]

        ### Embedding ###
        node_out = self.node_obs_embed(node_obs)
        pe = self.pe_embed(self.position_encodings.reshape(self.position_encodings.shape[0], -1).unsqueeze(0).repeat(x.shape[0], 1, 1))
        node_out = node_out + pe

        edge_out = self.edge_obs_embed(edge_obs)

        ### Attention ###
        for n, layer in enumerate(self.mhgat):
            node_out, edge_out, _ = layer(node_out, edge_out, None)

        ### Action Selection ###
        target_indices = [[i for _ in  range(self.latent_dim)] for i in target_index]
        target_indices = torch.tensor(target_indices, dtype=torch.int64).unsqueeze(axis=-1).reshape(len(target_index), 1, self.latent_dim).to(self.device)
        gather_out = torch.gather(node_out, 1, target_indices).squeeze(1).to(self.device)

        act = self.action_layer(gather_out)
        act = self.act_head(act)

        if bool(self.config['use_ppo']) == True:
            cri = self.critic_layer(gather_out)
        
            return act, cri
        else:
            return act
