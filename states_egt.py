import numpy as np
import math

from mdp_config import mdp_configs

reverse_direction = {
        'N' : 'S',
        'S' : 'N',
        'E' : 'W',
        'W' : 'E',
    }
left_turn_direction = {
    'N' : 'W',
    'S' : 'E',
    'E' : 'N',
    'W' : 'S',
}
right_turn_direction = {
    'N' : 'E',
    'S' : 'W',
    'E' : 'S',
    'W' : 'N',
}

def egt_node_state(signals):
    
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
            # obs.append(total_wait)
            # obs.append(total_speed)
            # obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)

    node_size = len(observations[list(observations.keys())[0]])

    return observations, node_size


def colight_edge_v0(signals):
    observations = dict()

    n_intersection = len(signals)
    
    observations, node_size = egt_node_state(signals)

    # Edge 결과물은 (node, node, edge_latent) shape,
    # downstream[direction] 으로 어느 노드에 들어갈지 알아낼 것,
    # node 순서는 signals 안의 순서로
    # 담는 정보는 lane의 총 차선수? + queue 이외 정보들

    signal_order = dict()
    for i, s in enumerate(list(observations.keys())):
        signal_order[s] = i
    reverse_direction = {
        'N' : 'S',
        'S' : 'N',
        'E' : 'W',
        'W' : 'E',
    }

    edge_size = 4
    edge_obs = np.zeros(shape=(len(signals), len(signals), max(edge_size, node_size)))

    for signal_id in signals:
        signal = signals[signal_id]
        unique_lane = {
            'N' : [],
            'S' : [],
            'E' : [],
            'W' : [],
        }

        for direction in signal.lane_sets:
            vehicle_head = direction[0]
            vehicle_src = reverse_direction[vehicle_head]
            # vehicle_dst = direction[1]

            edge_approach = 0
            edge_total_wait = 0
            edge_max_wait = 0

            lane_src = signal.downstream[vehicle_src]

            if lane_src is not None:

                for lane in signal.lane_sets[direction]:
                    if lane not in unique_lane[vehicle_src]:
                        unique_lane[vehicle_src].append(lane)

                        edge_obs[signal_order[signal_id], signal_order[lane_src], 1] += signal.full_observation[lane]['approach']
                        
                        edge_total_wait = signal.full_observation[lane]['total_wait']
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 2] += edge_total_wait

                        edge_max_wait = signal.full_observation[lane]['max_wait']
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 3] = \
                            max(edge_max_wait, edge_obs[signal_order[signal_id], signal_order[lane_src], 3])
                
                if edge_obs[signal_order[signal_id], signal_order[lane_src], 2] > 0:
                    # edge_obs[signal_order[signal_id], signal_order[lane_src], 2] = math.log(edge_obs[signal_order[signal_id], signal_order[lane_src], 2])
                    edge_obs[signal_order[signal_id], signal_order[lane_src], 2] = edge_obs[signal_order[signal_id], signal_order[lane_src], 2] / 28
                if edge_obs[signal_order[signal_id], signal_order[lane_src], 3] > 0:
                #     edge_obs[signal_order[signal_id], signal_order[lane_src], 3] = math.log(edge_obs[signal_order[signal_id], signal_order[lane_src], 3])
                    edge_obs[signal_order[signal_id], signal_order[lane_src], 3] = edge_obs[signal_order[signal_id], signal_order[lane_src], 3] / 28
                edge_obs[signal_order[signal_id], signal_order[lane_src], 0] = len(unique_lane[vehicle_src]) if len(unique_lane[vehicle_src]) > 0 else 0


    global_obs = np.array(list(observations.values()))
    if edge_size > node_size:
        zero_pad = np.zeros((n_intersection, edge_size - node_size))
        global_obs = np.concatenate([global_obs, zero_pad], axis=1)
    global_obs = global_obs.reshape(1, n_intersection, max(edge_size, node_size))

    index_info = np.zeros((1, global_obs.shape[1], max(edge_size, node_size)))
    index_info[0, 0, 0] = node_size
    index_info[0, 0, 1] = edge_size
    for i, signal_id in enumerate(signals):
        index_info[0, 0, 2] = i
        observations[signal_id] = np.concatenate([global_obs, edge_obs, index_info], axis=0)

    return observations


def colight_edge(signals):
    observations = dict()

    n_intersection = len(signals)
    
    observations, node_size = egt_node_state(signals)

    # Edge 결과물은 (node, node, edge_latent) shape,
    # downstream[direction] 으로 어느 노드에 들어갈지 알아낼 것,
    # node 순서는 signals 안의 순서로
    # 담는 정보는 lane의 총 차선수, 직진 approach, total wait, 좌회전, 우회전 순

    signal_order = dict()
    for i, s in enumerate(list(observations.keys())):
        signal_order[s] = i
    

    edge_size = 10
    edge_obs = np.zeros(shape=(len(signals), len(signals), max(edge_size, node_size)))

    for signal_id in signals:
        signal = signals[signal_id]
        unique_lane = {
            'N' : [],
            'S' : [],
            'E' : [],
            'W' : [],
        }

        for direction in signal.lane_sets:
            vehicle_head = direction[0]
            vehicle_src = reverse_direction[vehicle_head]
            vehicle_dst = direction[-1]

            lane_src = signal.downstream[vehicle_src]

            if lane_src is not None:

                # 담는 정보는 lane의 총 차선수 + approach, total wait, max_wait 의 직진, 좌회전, 우회전 순
                # 직진 1, 2, 3
                # 좌회전 4, 5, 6
                # 우회전 7, 8, 9
                for lane in signal.lane_sets[direction]:
                    if lane not in unique_lane[vehicle_src]:
                        unique_lane[vehicle_src].append(lane)

                    edge_total_wait = signal.full_observation[lane]['total_wait']
                    if edge_total_wait > 0:
                        edge_total_wait = math.log10(edge_total_wait)
                    edge_max_wait = signal.full_observation[lane]['max_wait']
                    if edge_max_wait > 0:
                        edge_max_wait = math.log10(edge_max_wait)

                    if vehicle_dst == vehicle_head:  # 직진
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 1] += signal.full_observation[lane]['approach']
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 2] += edge_total_wait
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 3] = \
                            max(edge_max_wait, edge_obs[signal_order[signal_id], signal_order[lane_src], 3])
                    elif vehicle_dst == left_turn_direction[vehicle_head]:  # 좌회전
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 4] += signal.full_observation[lane]['approach']
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 5] += edge_total_wait
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 6] = \
                            max(edge_max_wait, edge_obs[signal_order[signal_id], signal_order[lane_src], 6])
                    elif vehicle_dst == right_turn_direction[vehicle_head]:  # 우회전
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 7] += signal.full_observation[lane]['approach']
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 8] += edge_total_wait
                        edge_obs[signal_order[signal_id], signal_order[lane_src], 9] = \
                            max(edge_max_wait, edge_obs[signal_order[signal_id], signal_order[lane_src], 9])
                    else:
                        raise NotImplementedError
                
                edge_obs[signal_order[signal_id], signal_order[lane_src], 0] = len(unique_lane[vehicle_src]) if len(unique_lane[vehicle_src]) > 0 else 0


    global_obs = np.array(list(observations.values()))
    if edge_size > node_size:
        zero_pad = np.zeros((n_intersection, edge_size - node_size))
        global_obs = np.concatenate([global_obs, zero_pad], axis=1)
    global_obs = global_obs.reshape(1, n_intersection, max(edge_size, node_size))

    index_info = np.zeros((1, global_obs.shape[1], max(edge_size, node_size)))
    index_info[0, 0, 0] = node_size
    index_info[0, 0, 1] = edge_size
    for i, signal_id in enumerate(signals):
        index_info[0, 0, 2] = i
        observations[signal_id] = np.concatenate([global_obs, edge_obs, index_info], axis=0)

    return observations

def colight_edge_v2(signals):
    observations = dict()

    n_intersection = len(signals)

    observations, node_size = egt_node_state(signals)

    signal_order = dict()
    for i, s in enumerate(list(observations.keys())):
        signal_order[s] = i

    edge_size = 21
    edge_obs = np.zeros(shape=(len(signals), len(signals), max(edge_size, node_size)))

    for signal_id in signals:
        signal = signals[signal_id]
        lane_info = dict()
        for k, v in signal.lane_sets.items():
            for lane_id in v:
                edge_total_wait = signal.full_observation[lane]['total_wait']
                if edge_total_wait > 0:
                    edge_total_wait = math.log10(edge_total_wait)
                    
                edge_max_wait = signal.full_observation[lane]['max_wait']
                if edge_max_wait > 0:
                    edge_max_wait = math.log10(edge_max_wait)

                lane_info[lane_id] = [None, 0, signal.full_observation[lane_id]['approach'], edge_total_wait, edge_max_wait]

        for direction in signal.lane_sets:
            vehicle_head = direction[0]
            vehicle_src = reverse_direction[vehicle_head]
            vehicle_dst = direction[-1]

            lane_src = signal.downstream[vehicle_src]

            if lane_src is not None:
                for lane in signal.lane_sets[direction]:
                    lane_info[lane][0] = signal_order[lane_src]

                    if vehicle_dst == vehicle_head:                         lane_info[lane][1] += 1
                    elif vehicle_dst == left_turn_direction[vehicle_head]:  lane_info[lane][1] += 2
                    elif vehicle_dst == right_turn_direction[vehicle_head]: lane_info[lane][1] += 4
                    else:                                                   raise NotImplementedError

        for lane, lane_v in lane_info.items():
            if lane_v[0] is not None:
                pos = (lane_v[1] - 1) * 3
                edge_obs[signal_order[signal_id], lane_v[0], pos] += lane_v[2]  #approach
                edge_obs[signal_order[signal_id], lane_v[0], pos+1] += lane_v[3]  #total_wait
                edge_obs[signal_order[signal_id], lane_v[0], pos+2] = \
                    max(lane_v[4], edge_obs[signal_order[signal_id], lane_v[0], pos+2])  #max_wait

    global_obs = np.array(list(observations.values()))
    if edge_size > node_size:
        zero_pad = np.zeros((n_intersection, edge_size - node_size))
        global_obs = np.concatenate([global_obs, zero_pad], axis=1)
    global_obs = global_obs.reshape(1, n_intersection, max(edge_size, node_size))

    index_info = np.zeros((1, global_obs.shape[1], max(edge_size, node_size)))
    index_info[0, 0, 0] = node_size
    index_info[0, 0, 1] = edge_size
    for i, signal_id in enumerate(signals):
        index_info[0, 0, 2] = i
        observations[signal_id] = np.concatenate([global_obs, edge_obs, index_info], axis=0)

    return observations

def colight_edge_vnode(signals):
    observations = dict()

    n_intersection = len(signals)
    v_idx = n_intersection

    observations, node_size = egt_node_state(signals)

    signal_order = dict()
    for i, s in enumerate(list(observations.keys())):
        signal_order[s] = i

    edge_size = 21

    lane_info = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for k, v in signal.lane_sets.items():
            for lane_id in v:

                edge_total_wait = signal.full_observation[lane_id]['total_wait']
                if edge_total_wait > 0:
                    # edge_total_wait = math.log10(edge_total_wait)
                    edge_total_wait = edge_total_wait / 28
                    
                edge_max_wait = signal.full_observation[lane_id]['max_wait']
                if edge_max_wait > 0:
                    # edge_max_wait = math.log10(edge_max_wait)
                    edge_max_wait = edge_max_wait / 28

                lane_info[lane_id] = [signal_order[signal_id], None, 0, signal.full_observation[lane_id]['approach'], \
                                        edge_total_wait, edge_max_wait]

        lane_src_dict = {'S':None, 'W':None, "N":None, "E":None}
        for d in list(lane_src_dict.keys()):
            lane_from = signal.downstream[d]
            if lane_from is None:
                lane_from = v_idx
                v_idx += 1
            else:
                lane_from = signal_order[lane_from]
            lane_src_dict[d] = lane_from


        for direction in signal.lane_sets:
            vehicle_head = direction[0]
            vehicle_src = reverse_direction[vehicle_head]
            vehicle_dst = direction[-1]

            lane_src = lane_src_dict[vehicle_src]

            for lane in signal.lane_sets[direction]:
                lane_info[lane][1] = lane_src

                if vehicle_dst == vehicle_head:                         lane_info[lane][2] += 1
                elif vehicle_dst == left_turn_direction[vehicle_head]:  lane_info[lane][2] += 2
                elif vehicle_dst == right_turn_direction[vehicle_head]: lane_info[lane][2] += 4
                else:                                                   raise NotImplementedError

    edge_obs = np.zeros(shape=(v_idx, v_idx, max(edge_size, node_size)))
    for lane, lane_v in lane_info.items():
        pos = (lane_v[2] - 1) * 3
        edge_obs[lane_v[0], lane_v[1], pos] += lane_v[2]  #approach
        # edge_obs[lane_v[0], lane_v[1], pos+1] += math.log10(lane_v[3])  #total_wait
        # edge_obs[lane_v[0], lane_v[1], pos+2] = \
        #     max(math.log10(lane_v[4]), edge_obs[lane_v[0], lane_v[1], pos+2])  #max_wait

        edge_obs[lane_v[0], lane_v[1], pos+1] += lane_v[3] / 28  #total_wait
        edge_obs[lane_v[0], lane_v[1], pos+2] = \
            max(lane_v[4] / 28, edge_obs[lane_v[0], lane_v[1], pos+2])  #max_wait

    global_obs = np.array(list(observations.values()))
    if edge_size > node_size:
        zero_pad = np.zeros((n_intersection, edge_size - node_size))
        global_obs = np.concatenate([global_obs, zero_pad], axis=1)
    if v_idx > n_intersection:
        v_zero_pad = np.zeros((v_idx - n_intersection, max(edge_size, node_size)))
        global_obs = np.concatenate([global_obs, v_zero_pad], axis=0)
    global_obs = global_obs.reshape(1, max(n_intersection, v_idx), max(edge_size, node_size))

    index_info = np.zeros((1, global_obs.shape[1], max(edge_size, node_size)))
    index_info[0, 0, 0] = node_size
    index_info[0, 0, 1] = edge_size
    for i, signal_id in enumerate(signals):
        index_info[0, 0, 2] = i
        observations[signal_id] = np.concatenate([global_obs, edge_obs, index_info], axis=0)

    return observations