import rewards
import states

from agents.stochastic import STOCHASTIC
from agents.maxwave import MAXWAVE
from agents.maxpressure import MAXPRESSURE
from agents.pfrl_dqn import IDQN
from agents.pfrl_ppo import IPPO
from agents.pfrl_dppo import IDPPO
from agents.mplight import MPLight
from agents.fma2c import FMA2C
from agents.mplight_simf import MPLight_simf
from agents.mplight_noc import MPLight_noc
from agents.N1 import N1
from agents.attendlight import AttendLight
# from agents.colight_depre import CoLight
from agents.colight import CoLight
from agents.colight_lstm import CoLight_LSTM
from agents.colight_lstm_g import CoLight_LSTM_G
from agents.colight_edge import CoLight_edge
from agents.frap import IndependentFrap
from agents.advanced_mp import ADV_MP

from agents.pfrl_dqn_cycle import IDQN_cycle
from agents.pfrl_ppo_cycle import IPPO_cycle

agent_configs = {
    # *VAL configs have distance settings according to the validation scenarios
    'MAXWAVEVAL': {
        'agent': MAXWAVE,
        'state': states.wave,
        'reward': rewards.wait,
        'max_distance': 50
    },
    'MAXPRESSUREVAL': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 9999
    },
    'MPLightVAL': {
        'agent': MPLight,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 9999,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },
    'FMA2CVAL': {
        'agent': FMA2C,
        'state': states.fma2c,
        'reward': rewards.fma2c,
        'max_distance': 50,
        'management_acts': 4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 1e-5,
        'max_grad_norm': 40,
        'gamma': 0.96,
        'lr_init': 2.5e-4,
        'lr_decay': 'constant',
        'entropy_coef_init': 0.001,
        'entropy_coef_min': 0.001,
        'entropy_decay': 'constant',
        'entropy_ratio': 0.5,
        'value_coef': 0.5,
        'num_lstm': 64,
        'num_fw': 128,
        'num_ft': 32,
        'num_fp': 64,
        'batch_size': 120,
        'reward_norm': 2000.0,
        'reward_clip': 2.0,
    },

    'STOCHASTIC': {
        'agent': STOCHASTIC,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 1
    },
    'MAXWAVE': {
        'agent': MAXWAVE,
        'state': states.wave,
        'reward': rewards.wait,
        'max_distance': 50
    },
    'MAXPRESSURE': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 200
    },
    'ADV_MP': {
        'agent': ADV_MP,
        'state': states.adv_pressure,
        'reward': rewards.wait,
        'max_distance': 200
    },
    'IDQN': {
        'agent': IDQN,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        # 'state': states.mplight_full,
        # 'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500
    },
    'IDQN_cycle': {
        'agent': IDQN_cycle,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        # 'state': states.mplight_full,
        # 'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500
    },
    'IPPO': {
        'agent': IPPO,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200
    },
    'IPPO_cycle': {
        'agent': IPPO_cycle,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200
    },
    'IDPPO': {
        'agent': IDPPO,
        'state': states.drq_norm_neighbor,
        'reward': rewards.wait_norm_neighbor,
        'max_distance': 200
    },
    'FRAP': {
        'agent': IndependentFrap,
        'state': states.mplight,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },
    'MPLight': {
        'agent': MPLight,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },
    'MPLight_noc': {
        'agent': MPLight_noc,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },
    'MPLight_simf': {
        'agent': MPLight_simf,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },
    'MPLight_adv': {
        'agent': MPLight,
        'state': states.adv_pressure,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 2
    },
    'N1': {
        'agent': N1,
        'state': states.mplight,
        'reward': rewards.pressure_with_neighbor,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    },
    'FMA2C': {
        'agent': FMA2C,
        'state': states.fma2c,
        'reward': rewards.fma2c,
        'max_distance': 200,
        'management_acts': 4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 1e-5,
        'max_grad_norm': 40,
        'gamma': 0.96,
        'lr_init': 2.5e-4,
        'lr_decay': 'constant',
        'entropy_coef_init': 0.001,
        'entropy_coef_min': 0.001,
        'entropy_decay': 'constant',
        'entropy_ratio': 0.5,
        'value_coef': 0.5,
        'num_lstm': 64,
        'num_fw': 128,
        'num_ft': 32,
        'num_fp': 64,
        'batch_size': 120,
        'reward_norm': 2000.0,
        'reward_clip': 2.0,
    },

    # *FULL configs extend state space to include obs. available to IDQN
    'MPLightFULL': {
        'agent': MPLight,
        'state': states.mplight_full,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4
    },
    'AttendLight': {
        'agent': AttendLight,
        'state': states.mplight_full,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4
    },
    'FMA2CFULL': {
        'agent': FMA2C,
        'state': states.fma2c_full,
        'reward': rewards.fma2c_full,
        'max_distance': 200,
        'management_acts': 4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 1e-5,
        'max_grad_norm': 40,
        'gamma': 0.96,
        'lr_init': 2.5e-4,
        'lr_decay': 'constant',
        'entropy_coef_init': 0.001,
        'entropy_coef_min': 0.001,
        'entropy_decay': 'constant',
        'entropy_ratio': 0.5,
        'value_coef': 0.5,
        'num_lstm': 64,
        'num_fw': 128,
        'num_ft': 32,
        'num_fp': 64,
        'batch_size': 120,
        'reward_norm': 2000.0,
        'reward_clip': 2.0,
    },
    'CoLight_history_g': {
        'agent': CoLight_LSTM_G,
        'state': states.colight_history,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4,

        'N_HEAD' : 4,
        'N_DIM' : 16,

        'use_ppo' : False,
    },
    'CoLight_history': {
        'agent': CoLight_LSTM,
        'state': states.colight_history,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4,

        'N_HEAD' : 4,
        'N_DIM' : 16,

        'use_ppo' : False,
    },
    'CoLight_adv': {
        'agent': CoLight,
        'state': states.colight_adv,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4,

        'N_HEAD' : 4,
        'N_DIM' : 16,

        'use_ppo' : False,
    },
    'CoLight': {
        'agent': CoLight,
        'state': states.colight_state,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4,

        'N_HEAD' : 4,
        'N_DIM' : 16,

        'use_ppo' : False,
    },
    'CoLight_edge': {
        'agent': CoLight_edge,
        'state': states.colight_edge_v0,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4,

        'N_HEAD' : 4,
        'N_DIM' : 16,
        
        'use_vnode' : False,
        'use_ppo' : False,
    },
    'CoLight_edge_vnode': {
        'agent': CoLight_edge,
        'state': states.colight_edge_vnode,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 4,

        'N_HEAD' : 4,
        'N_DIM' : 16,

        'use_vnode' : True,
        'use_ppo' : False,
    },
}
