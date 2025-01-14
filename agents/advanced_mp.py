import numpy as np
from agents.agent import SharedAgent, Agent
from signal_config import signal_configs


class ADV_MP(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        self.map_name = map_name
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.demand_weight = 0.2

        self.agent = ADV_MPAGENT(self.map_name, signal_configs[map_name]['phase_pairs'], self.demand_weight)


class ADV_MPAGENT(Agent):
    def __init__(self, map_name, phase_pairs, demand_weight):
        super().__init__()
        self.map_name = map_name
        self.phase_pairs = phase_pairs
        self.demand_weight = demand_weight

    def act(self, observations, valid_acts=None, reverse_valid=None):
        acts = []
        for i, observation in enumerate(observations):
            cur_phase = int(observation[0])
            obs = observation[1:]

            if self.map_name == 'ingolstadt21' and i == 11 and cur_phase == 3:
                cur_phase = 2

            if valid_acts is None:
                all_press = []
                for phase, pair in enumerate(self.phase_pairs):
                    if phase == cur_phase:
                        all_press.append((obs[(pair[0]*2)+1] + obs[(pair[1]*2)+1]) * self.demand_weight)
                    else:
                        all_press.append(obs[pair[0] * 2] + obs[pair[1] * 2])
                acts.append(np.argmax(all_press))
            else:
                max_press, max_index = None, None
                for idx in valid_acts[i]:
                    pair = self.phase_pairs[idx]
                    if reverse_valid[i][cur_phase] == idx:
                        press = (obs[(pair[0]*2)+1] + obs[(pair[1]*2)+1]) * self.demand_weight
                        # press = obs[pair[0] * 2] + obs[pair[1] * 2]
                    else:
                        press = obs[pair[0] * 2] + obs[pair[1] * 2]
                    if max_press is None:
                        max_press = press
                        max_index = idx
                    if press > max_press:
                        max_press = press
                        max_index = idx
                acts.append(valid_acts[i][max_index])
        return acts

    def observe(self, observation, reward, done, info):
        pass

    def save(self, path):
        pass