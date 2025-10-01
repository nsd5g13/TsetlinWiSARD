import random
import numpy as np

def LUT(datain, sel):
	in_width = len(datain)
	if 2**in_width == len(sel):
		datain_int = int(datain, 2)
		dataout = sel[datain_int]
	else:
		print("Unmatched LUT input width (%d-bit) and select signal (%d-bit)" %(in_width, len(sel)))
	return dataout

def reinforcement_1layer(states, no_state, prob, datain, true_label, predict_label):
	rand_numbers = [random.uniform(0,1) for i in range(2*len(states[true_label]))]
	middle_state = int(no_state/2)
	datain_int = datain.dot(2**np.arange(datain.shape[1]-1,-1,-1))
	for i in range(len(states[true_label])):
		index = datain_int[i]
		if rand_numbers[i] < prob:
			if states[true_label][i][index] != no_state-1:
				states[true_label][i][index] = states[true_label][i][index] + 1
		if rand_numbers[i+len(states[true_label])] < prob:
			if states[predict_label][i][index] != 0:
				states[predict_label][i][index] = states[predict_label][i][index] - 1
	return states

def reward(states, no_state, prob, datain):
	reward_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i, LUT_states in enumerate(states):
		index = datain_int[i]
		state = states[i][index]
		if reward_rand[i] < prob:
			if (state < middle_state and state != 0):
				states[i][index] = states[i][index] - 1
			elif (state >= middle_state and state != no_state-1):
				states[i][index] = states[i][index] + 1
	return states

def reward_all_1layer(states, no_state, prob, datain):
	reward_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	#datain_int = []
	datain_int = datain.dot(2**np.arange(datain.shape[1]-1,-1,-1))
	#for each in datain:
	#	each = [str(x) for x in each]
	#	each_int = int(''.join(each), 2)
	#	datain_int.append(each_int)
	for i in range(len(states)):
		index = datain_int[i]
		if reward_rand[i] < prob:
			if states[i][index] != no_state-1:
				states[i][index] = states[i][index] + 1
	return states

def reward_only0_1layer(states, no_state, prob, datain, LUTouts):
	reward_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i in range(len(states)):
		index = datain_int[i]
		if reward_rand[i] < prob and LUTouts[i] == 0:
			if states[i][index] != no_state-1:
				states[i][index] = states[i][index] + 1
	return states

def reward_all_multilayer(states, no_state, prob, datain):
	reward_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i, LUT_states in enumerate(states):
		index = datain_int[i]
		state = states[i][index]
		if reward_rand[i] < prob:
			if states[i][index] != no_state-1:
				states[i][index] = states[i][index] + 1
	return states

def reward_bak0303(states, no_state, prob, datain):
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i, LUT_states in enumerate(states):
		reward_rand = random.uniform(0,1)
		index = datain_int[i]
		state = states[i][index]
		if reward_rand < prob:
			if (state < middle_state and state != 0):
				states[i][index] = states[i][index] - 1
			elif (state >= middle_state and state != no_state-1):
				states[i][index] = states[i][index] + 1
	return states

def reward_bak(states, no_state, prob, seed):
	#random.seed(seed)
	middle_state = int(no_state/2)
	for i, LUT_states in enumerate(states):
		for j, state in enumerate(LUT_states):
			reward_rand = random.uniform(0,1)
			if reward_rand < prob:
				if (state < middle_state and state != 0):
					states[i][j] = states[i][j] - 1
				elif (state >= middle_state and state != no_state-1):
					states[i][j] = states[i][j] + 1
	return states

def penalty(states, no_state, prob, datain):
	penalty_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i, LUT_states in enumerate(states):
		index = datain_int[i]
		state = states[i][index]
		if penalty_rand[i] < prob:
			if (state < middle_state or state == 0):
				states[i][index] = states[i][index] + 1
			elif (state >= middle_state or state == no_state-1):
				states[i][index] = states[i][index] - 1
	return states

def penalty_all_1layer(states, no_state, prob, datain):
	penalty_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	#datain_int = []
	datain_int = datain.dot(2**np.arange(datain.shape[1]-1,-1,-1))
	#for each in datain:
	#	each = [str(x) for x in each]
	#	each_int = int(''.join(each), 2)
	#	datain_int.append(each_int)
	for i in range(len(states)):
		index = datain_int[i]
		if penalty_rand[i] < prob:
			if states[i][index] != 0:
				states[i][index] = states[i][index] - 1
	return states

def penalty_only1_1layer(states, no_state, prob, datain, LUTouts):
	penalty_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i in range(len(states)):
		index = datain_int[i]
		if penalty_rand[i] < prob and LUTouts[i] == 1:
			if states[i][index] != 0:
				states[i][index] = states[i][index] - 1
	return states

def penalty_all_multilayer(states, no_state, prob, datain):
	penalty_rand = [random.uniform(0,1) for i in range(len(states))]
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i, LUT_states in enumerate(states):
		index = datain_int[i]
		state = states[i][index]
		if penalty_rand[i] < prob:
			if states[i][index] != 0:
				states[i][index] = states[i][index] - 1
	return states

def penalty_bak0303(states, no_state, prob, datain):
	middle_state = int(no_state/2)
	datain_int = []
	for each in datain:
		each = [str(x) for x in each]
		each_int = int(''.join(each), 2)
		datain_int.append(each_int)
	for i, LUT_states in enumerate(states):
		reward_rand = random.uniform(0,1)
		index = datain_int[i]
		state = states[i][index]
		if reward_rand < prob:
			if (state < middle_state or state == 0):
				states[i][index] = states[i][index] + 1
			elif (state >= middle_state or state == no_state-1):
				states[i][index] = states[i][index] - 1
	return states

def penalty_bak(states, no_state, prob, seed):
	#random.seed(seed)
	middle_state = int(no_state/2)
	for i, LUT_states in enumerate(states):
		for j, state in enumerate(LUT_states):
			penalty_rand = random.uniform(0,1)
			if penalty_rand < prob:
				if (state < middle_state or state == 0):
					states[i][j] = states[i][j] + 1
				elif (state >= middle_state or state == no_state-1):
					states[i][j] = states[i][j] - 1
	return states

def relax(states, no_state, degree):
	middle_state = int(no_state/2)
	for i, LUT_states in enumerate(states):
		for j , state in enumerate(LUT_states):
			if state < middle_state:
				states[i][j] = int(state + np.floor((middle_state-state)/degree))
			else:
				states[i][j] = int(state - np.floor((state-middle_state)/degree))
	return states

def initilization(no_state, no_LUTinputs, no_LUTs, seed):
	random.seed(seed)
	middle_state = int(no_state/2)
	no_layers = len(no_LUTs)
	TA_states = [[] for i in range(no_layers)]
	for i in range(no_layers):
		no_LUTs_layer = no_LUTs[i]
		no_LUTinputs_layer = no_LUTinputs[i]
		for j in range(no_LUTs_layer):
			TA_states[i].append([middle_state - random.randint(0,1) for i in range(2**no_LUTinputs_layer)])
	return TA_states

def initilization_1layer(no_state, no_LUTinputs, no_LUTs, seed):
	random.seed(seed)
	middle_state = int(no_state/2)
	TA_states = []
	for j in range(no_LUTs):
		TA_states.append([middle_state - random.randint(0,1) for i in range(2**no_LUTinputs)])
		#TA_states.append([0 for i in range(2**no_LUTinputs)])
	return TA_states

def random_connection(no_LUTinputs, no_LUTs, seed):
	random.seed(seed)
	connections = [] # [layer_idx, LUT_idx (output_idx), LUT_idx@next layer, input_idx] 
	no_layers = len(no_LUTs)
	for i in range(no_layers-1):
		no_LUTs_ThisLayer = no_LUTs[i]
		no_LUTs_NextLayer = no_LUTs[i+1]
		no_LUTinputs_NextLayer = no_LUTinputs[i+1]

		if no_LUTs_ThisLayer == no_LUTs_NextLayer * no_LUTinputs_NextLayer:
			for j in range(no_LUTs_ThisLayer):
				connections.append([i, j, int(np.floor(j/no_LUTinputs_NextLayer)), j%no_LUTinputs_NextLayer])
		else:
			for j in range(no_LUTs_NextLayer):
				LUT_connections = random.sample(range(no_LUTs_ThisLayer), no_LUTinputs_NextLayer)
				for k, each in enumerate(LUT_connections):
					connections.append([i, each, j, k])

	return connections

def random_connection_bypass(no_LUTinputs, no_LUTs, seed, bypass, no_features):
	random.seed(seed)
	connections = [] # [layer_idx, LUT_idx (output_idx), LUT_idx@next layer, input_idx, raw_feature_idx] 
	no_layers = len(no_LUTs)
	for i in range(no_layers-1):
		no_LUTs_ThisLayer = no_LUTs[i]
		no_LUTs_NextLayer = no_LUTs[i+1]
		no_LUTinputs_NextLayer = no_LUTinputs[i+1]
		bypass_NextLayer = bypass[i+1]

		# without bypassing inputs
		if bypass_NextLayer == 0:
			if no_LUTs_ThisLayer == no_LUTs_NextLayer * no_LUTinputs_NextLayer:
				for j in range(no_LUTs_ThisLayer):
					connections.append([i, j, int(np.floor(j/no_LUTinputs_NextLayer)), j%no_LUTinputs_NextLayer, 'x'])
			else:
				for j in range(no_LUTs_NextLayer):
					LUT_connections = random.sample(range(no_LUTs_ThisLayer), no_LUTinputs_NextLayer)
					for k, each in enumerate(LUT_connections):
						connections.append([i, each, j, k, 'x'])

		# with bypassing inputs from the input features
		else:
			for j in range(no_LUTs_NextLayer):
				for k in range(no_LUTinputs_NextLayer):
					rand_value = random.uniform(0,1)
					#p_RawFeature = float(bypass_NextLayer/(bypass_NextLayer+no_LUTs_ThisLayer))
					p_RawFeature = float(bypass_NextLayer/(no_LUTs_NextLayer * no_LUTinputs_NextLayer))

					if rand_value <= p_RawFeature: # bypass raw features
						raw_feature_idx = random.randint(0, no_features-1)
						connections.append([i, 'x', j, k, raw_feature_idx])

					else: # input of next layer comes from the previous layer
						LUT_idx = random.randint(0, no_LUTs_ThisLayer-1)
						connections.append([i, LUT_idx, j, k, 'x'])

	return connections

def predict(x, connections, no_LUTinputs, no_LUTs, TA_states, no_state):
	all_datain = []
	middle_state = int(no_state/2)
	no_layers = len(no_LUTs)
	for i in range(no_layers):
		no_LUTs_ThisLayer = no_LUTs[i]
		no_LUTinputs_ThisLayer = no_LUTinputs[i]
		TA_states_ThisLayer = TA_states[i]

		if i == 0:
			datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		else:
			datain = np.reshape(datain_NextLayer, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		all_datain.append(datain)
		
		# outputs of current layer		
		dataout_ThisLayer = []		
		for j in range(len(datain)):
			LayerIn = datain[j].astype('str')
			LayerIn = ''.join(LayerIn)
			sel = [int(s>=middle_state) for s in TA_states_ThisLayer[j]]
			LayerOut = LUT(LayerIn, sel)
			dataout_ThisLayer.append(LayerOut)

		# inputs of next layer
		datain_NextLayer = []
		if i == no_layers-1:
			predict_label = dataout_ThisLayer
		else:
			connections_idx = [connections.index(each_connections) for each_connections in connections if each_connections[0] == i]
			connections_ThisLayer = [connections[idx] for idx in connections_idx]
			for each_connection in connections_ThisLayer:
				datain_NextLayer.append(dataout_ThisLayer[each_connection[1]])

	return predict_label, all_datain

def predict_bypass(x, connections, no_LUTinputs, no_LUTs, TA_states, no_state, x_raw):
	all_datain = []
	middle_state = int(no_state/2)
	no_layers = len(no_LUTs)
	for i in range(no_layers):
		no_LUTs_ThisLayer = no_LUTs[i]
		no_LUTinputs_ThisLayer = no_LUTinputs[i]
		TA_states_ThisLayer = TA_states[i]

		if i == 0:
			datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		else:
			datain = np.reshape(datain_NextLayer, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		all_datain.append(datain)
		
		# outputs of current layer		
		dataout_ThisLayer = []		
		for j in range(len(datain)):
			LayerIn = datain[j].astype('str')
			LayerIn = ''.join(LayerIn)
			sel = [int(s>=middle_state) for s in TA_states_ThisLayer[j]]
			LayerOut = LUT(LayerIn, sel)
			dataout_ThisLayer.append(LayerOut)

		# inputs of next layer
		datain_NextLayer = []
		if i == no_layers-1:
			predict_label = dataout_ThisLayer
		else:
			connections_idx = [connections.index(each_connections) for each_connections in connections if each_connections[0] == i]
			connections_ThisLayer = [connections[idx] for idx in connections_idx]
			for each_connection in connections_ThisLayer:
				if each_connection[1] != 'x': # input of next layer comes from the previous layer
					datain_NextLayer.append(dataout_ThisLayer[each_connection[1]])
				else:	# bypass raw features
					feature_idx = each_connection[-1]
					datain_NextLayer.append(x_raw[feature_idx])

	return predict_label, all_datain

def predict_1layer(x, no_LUTinputs, no_LUTs, TA_states, no_state):
	middle_state = int(no_state/2)

	no_LUTs_ThisLayer = no_LUTs
	no_LUTinputs_ThisLayer = no_LUTinputs
	TA_states_ThisLayer = TA_states

	datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		
	# outputs of current layer		
	dataout = []		
	for j in range(len(datain)):
		LayerIn = datain[j].astype('str')
		LayerIn = ''.join(LayerIn)
		sel = [int(s>=middle_state) for s in TA_states_ThisLayer[j]]
		LayerOut = LUT(LayerIn, sel)
		dataout.append(LayerOut)
	ClassSum = sum(dataout)

	return dataout, ClassSum

def predict_multi_layer(x, connections, no_LUTinputs, no_LUTs, TA_states, no_state):
	all_datain = []
	middle_state = int(no_state/2)
	no_layers = len(no_LUTs)
	for i in range(no_layers):
		no_LUTs_ThisLayer = no_LUTs[i]
		no_LUTinputs_ThisLayer = no_LUTinputs[i]
		TA_states_ThisLayer = TA_states[i]

		if i == 0:
			datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		else:
			datain = np.reshape(datain_NextLayer, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		all_datain.append(datain)
		
		# outputs of current layer		
		dataout_ThisLayer = []		
		for j in range(len(datain)):
			LayerIn = datain[j].astype('str')
			LayerIn = ''.join(LayerIn)
			sel = [int(s>=middle_state) for s in TA_states_ThisLayer[j]]
			LayerOut = LUT(LayerIn, sel)
			dataout_ThisLayer.append(LayerOut)

		# inputs of next layer
		datain_NextLayer = []
		if i == no_layers-1:
			ClassSum = sum(dataout_ThisLayer)
		else:
			connections_idx = [connections.index(each_connections) for each_connections in connections if each_connections[0] == i]
			connections_ThisLayer = [connections[idx] for idx in connections_idx]
			for each_connection in connections_ThisLayer:
				datain_NextLayer.append(dataout_ThisLayer[each_connection[1]])

	return ClassSum, all_datain	

def confidence_level(TA_states, all_LayerIn):
	all_confidence = []
	for TA_states_layer, LayerIn in zip(TA_states, all_LayerIn):
		for TA_states_LUT, LUTIn in zip(TA_states_layer, LayerIn):
			LUTIn = LUTIn.astype('str')
			LUTIn = ''.join(LUTIn)
			LUTIn_int = int(LUTIn, 2)
			confidence = TA_states_LUT[LUTIn_int]
			all_confidence.append(confidence)
	return all_confidence

def wisard(x, no_LUTinputs, no_LUTs, LUTouts, true_positive):
	no_LUTs_ThisLayer = no_LUTs
	no_LUTinputs_ThisLayer = no_LUTinputs
	new_LUTouts = LUTouts

	datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
		
	# outputs of current layer		
	dataout = []		
	for j in range(len(datain)):
		LayerIn = datain[j].astype('str')
		LayerIn = ''.join(LayerIn)
		sel = [int(s) for s in new_LUTouts[j]]
		LayerOut = LUT(LayerIn, sel)
		dataout.append(LayerOut)
		if true_positive == True:
			new_LUTouts[j][int(LayerIn,2)] = 1
	ClassSum = sum(dataout)

	return new_LUTouts, ClassSum

def bleaching(x, no_LUTinputs, no_LUTs, LUTouts, true_positive):
	no_LUTs_ThisLayer = no_LUTs
	no_LUTinputs_ThisLayer = no_LUTinputs
	new_LUTouts = LUTouts

	datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))
				
	for j in range(len(datain)):
		LayerIn = datain[j].astype('str')
		LayerIn = ''.join(LayerIn)
		if true_positive == True:
			new_LUTouts[j][int(LayerIn,2)] = new_LUTouts[j][int(LayerIn,2)] + 1

	return new_LUTouts

def bleaching_response(x, no_LUTinputs, no_LUTs, LUTouts, b):
	no_LUTs_ThisLayer = no_LUTs
	no_LUTinputs_ThisLayer = no_LUTinputs
	datain = np.reshape(x, (no_LUTs_ThisLayer, no_LUTinputs_ThisLayer))

	# outputs of current layer		
	dataout = []
	for j in range(len(datain)):
		LayerIn = datain[j].astype('str')
		LayerIn = ''.join(LayerIn)
		sel = [int(s>b) for s in LUTouts[j]]
		LayerOut = LUT(LayerIn, sel)
		dataout.append(LayerOut)
	ClassSum = sum(dataout)
	
	return ClassSum

		