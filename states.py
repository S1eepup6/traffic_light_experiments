import numpy as np
import math

from mdp_config import mdp_configs

from states_egt import *

def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.phase
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            lane_obs.append(signal.full_observation[lane]['approach'])
            lane_obs.append(signal.full_observation[lane]['total_wait'])
            lane_obs.append(signal.full_observation[lane]['queue'])

            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += vehicle['speed']
            lane_obs.append(total_speed)

            obs.append(lane_obs)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm_helper(signal):
    obs = []
    act_index = signal.phase
    for i, lane in enumerate(signal.lanes):
        lane_obs = []
        if i == act_index:
            lane_obs.append(1)
        else:
            lane_obs.append(0)

        lane_obs.append(signal.full_observation[lane]['approach'] / 28)
        lane_obs.append(signal.full_observation[lane]['total_wait'] / 28)
        lane_obs.append(signal.full_observation[lane]['queue'] / 28)

        total_speed = 0
        vehicles = signal.full_observation[lane]['vehicles']
        for vehicle in vehicles:
            total_speed += (vehicle['speed'] / 20 / 28)
        lane_obs.append(total_speed)

        obs.append(lane_obs)
    return obs

def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = drq_norm_helper(signal)
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations

def drq_norm_neighbor(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        local_obs = drq_norm_helper(signal)
        obs = np.expand_dims(np.asarray(local_obs), axis=0)

        neighbor_obs = None
        for key in signal.downstream:
            neighbor_id = signal.downstream[key]
            if neighbor_id is not None:
                neighbor = signals[neighbor_id]
                neighbor_obs_tmp = drq_norm_helper(neighbor)
                neighbor_obs_tmp = np.expand_dims(np.asarray(neighbor_obs_tmp), axis=0)

                if neighbor_obs is None:
                    neighbor_obs = neighbor_obs_tmp
                else:
                    neighbor_obs = np.concatenate((neighbor_obs, neighbor_obs_tmp), axis=1)
                # obs = np.concatenate((obs, neighbor_obs), axis=1)

        obs = [obs, neighbor_obs]

        observations[signal_id] = obs

    return observations

def adv_pressure(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        if signal_id == "243641585" and signal.phase == 5:
            obs = [2]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
            if len(signal.lane_sets[direction]) > 0:
                queue_length /= len(signal.lane_sets[direction])
            
            # Subtract downstream
            n_dwn_lane = 0
            n_dwn_veh = 0
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    n_dwn_veh += signal.signals[dwn_signal].full_observation[lane]['queue']
                    n_dwn_lane += 1
            if n_dwn_lane > 0:
                # queue_length -= n_dwn_veh
                queue_length -= n_dwn_veh / n_dwn_lane

            # Effective Running 
            n_effective_running_veh = 0
            for lane in signal.lane_sets[direction]:
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    if vehicle['distance'] < 110:
                        n_effective_running_veh += 1

            obs.append(queue_length)
            obs.append(n_effective_running_veh)

        observations[signal_id] = np.asarray(obs)
    return observations

def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        if signal_id == "243641585" and signal.phase == 5:
            obs = [2]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]

        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            total_wait = 0
            total_speed = 0
            tot_approach = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
                total_wait += (signal.full_observation[lane]['total_wait'] / 28)
                total_speed = 0
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    total_speed += vehicle['speed']
                tot_approach += (signal.full_observation[lane]['approach'] / 28)

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].full_observation[lane]['queue']
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations

def wave(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        state = []
        for direction in signal.lane_sets:
            wave_sum = 0
            for lane in signal.lane_sets[direction]:
                wave_sum += signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            state.append(wave_sum)
        observations[signal_id] = np.asarray(state)
    return observations


def ma2c(signals):
    ma2c_config = mdp_configs['MA2C']

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / ma2c_config['norm_wave'], 0, ma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(ma2c_config['coop_gamma'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / ma2c_config['norm_wait'], 0, ma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    fma2c_config = mdp_configs['FMA2C']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    fma2c_config = mdp_configs['FMA2CFull']
    management = fma2c_config['management']
    supervisors = fma2c_config['supervisors']   # reverse of management
    management_neighbors = fma2c_config['management_neighbors']

    region_fringes = dict()
    for manager in management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in management_neighbors[manager]:
            neighborhood.append(fma2c_config['alpha'] * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            wave = signal.full_observation[lane]['queue'] + signal.full_observation[lane]['approach']
            waves.append(wave)

            waves.append(signal.full_observation[lane]['total_wait'] / 28)
            total_speed = 0
            vehicles = signal.full_observation[lane]['vehicles']
            for vehicle in vehicles:
                total_speed += (vehicle['speed'] / 20 / 28)
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(np.asarray(waves) / fma2c_config['norm_wave'], 0, fma2c_config['clip_wave'])

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                waves.append(fma2c_config['alpha'] * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.full_observation[lane]['max_wait']
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / fma2c_config['norm_wait'], 0, fma2c_config['clip_wait'])

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations

def colight_state(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']

            obs.append(queue_length)
        observations[signal_id] = np.asarray(obs)
    
    obs_size = len(observations[list(observations.keys())[0]])

    global_obs = list(observations.values())
    for i, signal_id in enumerate(signals):
        observations[signal_id] = np.array(global_obs + [[i] * obs_size])

    return observations

def colight_history(signals):
    observations = dict()
    _signal_id_test = list(signals.keys())[0]

    for signal_id in signals:
        observations[signal_id] = list()

    len_history = len(signals[list(signals.keys())[0]].history)
    # print(len_history)

    for i in range(len_history):
        obs_i = dict()

        for signal_id in signals:
            signal = signals[signal_id]
            obs = [signal.phase]
            for direction in signal.lane_sets:
                # Add inbound
                queue_length = 0
                for lane in signal.lane_sets[direction]:
                    queue_length += signal.full_observation[lane]['queue']

                obs.append(queue_length)
            obs_i[signal_id] = np.asarray(obs)
        
        obs_size = len(obs_i[list(obs_i.keys())[0]])

        global_obs = list(obs_i.values())
        for i, signal_id in enumerate(signals):
            observations[signal_id].append( global_obs + [[i] * obs_size] )

    for i, signal_id in enumerate(signals):
        observations[signal_id] = np.array( observations[signal_id] )

    # print(observations[_signal_id_test].shape)

    return observations

def colight_adv(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.phase]
        if signal_id == "243641585" and signal.phase == 5:
            obs = [2]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.full_observation[lane]['queue']
            if len(signal.lane_sets[direction]) > 0:
                queue_length /= len(signal.lane_sets[direction])

            # Subtract downstream
            n_dwn_lane = 0
            n_dwn_veh = 0
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signalid[lane]
                if dwn_signal in signal.signals:
                    n_dwn_veh += signal.signals[dwn_signal].full_observation[lane]['queue']
                    n_dwn_lane += 1
            if n_dwn_lane > 0:
                queue_length -= n_dwn_veh / n_dwn_lane

            # Effective Running 
            n_effective_running_veh = 0
            for lane in signal.lane_sets[direction]:
                vehicles = signal.full_observation[lane]['vehicles']
                for vehicle in vehicles:
                    if vehicle['distance'] < 110:
                        n_effective_running_veh += 1

            obs.append(queue_length)
            obs.append(n_effective_running_veh)

        observations[signal_id] = np.asarray(obs)
    
    obs_size = len(observations[list(observations.keys())[0]])

    global_obs = list(observations.values())
    for i, signal_id in enumerate(signals):
        observations[signal_id] = np.array(global_obs + [[i] * obs_size])

    return observations