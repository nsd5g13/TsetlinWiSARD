import WNN
import numpy as np
import random
from sklearn.utils import shuffle
from keras.datasets import mnist
import time
import multiprocessing
import sys

print("Number of cpu : ", multiprocessing.cpu_count())
pool = multiprocessing.Pool(processes=10)

# model
no_LUTinputs = int(sys.argv[1])
no_LUTs = int(sys.argv[2])

# dataset
dataset = sys.argv[3]
X_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_test.npy')
no_class = len(set(Y_train))
no_features = len(X_train[0])
LUTouts_classes = [[[0 for k in range(2**no_LUTinputs)] for j in range(no_LUTs)] for i in range(no_class)]
no_state = 2**8
middle_state = no_state/2
prob = 0.5

print('# %s dimensions: %d training samples, %d test samples, %d classes, %d Boolean features' %(dataset, len(Y_train), len(Y_test), no_class, no_features))

no_epoch = int(sys.argv[4])
thread = sys.argv[5]
	
TA_states_classes = []
seed = 10
for i in range(no_class):
	TA_states = WNN.initilization_1layer(no_state, no_LUTinputs, no_LUTs, seed)
	TA_states_classes.append(TA_states)

# input connections
no_extra = int(np.ceil(no_features/no_LUTinputs) * no_LUTinputs - no_features)
extra_features = np.zeros((len(X_train), no_extra), dtype=int)
X_train = np.concatenate((X_train, extra_features), axis=1)
extra_features = np.zeros((len(X_test), no_extra), dtype=int)
X_test = np.concatenate((X_test, extra_features), axis=1)
no_inputs_layer0 = no_LUTinputs * no_LUTs
seed = 1
random.seed(seed)
input_connections = []
if no_inputs_layer0%(no_features+no_extra) != 0:
	duplicates = int(np.ceil(no_inputs_layer0/(no_features+no_extra)))
else:
	duplicates = int(np.ceil(no_inputs_layer0/(no_features+no_extra)))+1
for i in range(no_class):
	random.seed(seed)
	input_connections.append([])
	if duplicates != 1:
		for j in range(duplicates-1):
			input_connections[i].extend(random.sample(range(no_features+no_extra), no_features+no_extra))
			#input_connections[i].extend(range(no_features+no_extra))
		duplicates_floor = int(np.floor(no_inputs_layer0/(no_features+no_extra)))
		if duplicates != duplicates_floor:
			input_connections[i].extend(random.sample(range(no_features+no_extra), no_inputs_layer0-duplicates_floor*(no_features+no_extra)))
			#input_connections[i].extend(range(no_features+no_extra)[0:no_inputs_layer0-duplicates_floor*(no_features+no_extra)])
	else:
		input_connections[i].extend(random.sample(range(no_features+no_extra), no_features+no_extra))
		#input_connections[i].extend(range(no_features+no_extra))
	

f = open("accuracy_1layer.txt","w")

train_prob = [1 for i in range(no_class)]

#TA_states_classes = np.load(r'TA_states_40.npy')
#TA_states_classes = TA_states_classes.tolist()

for i in range(no_epoch):
	# training
	no_trainsamp = 0
	no_correct = 0

	no_trainsamp_class = [1 for i in range(no_class)]
	no_correct_class = [0 for i in range(no_class)]

	# shuffle training samples
	#XY_train = np.concatenate((X_train, np.array([Y_train.tolist()]).T), axis=1)
	#np.random.seed(seed+i)
	#np.random.shuffle(XY_train)
	#X_train_shuffle = XY_train[:,:-1]
	#Y_train_shuffle = XY_train[:,-1]

	t_start = time.time()
	for x, y in zip(X_train, Y_train):
		no_trainsamp = no_trainsamp + 1

		#LUTouts_class = []
		ClassSum_class = []

		if thread == 'single':
			# single thread
			for k in range(no_class):
				x_connections = np.array([x[j] for j in input_connections[k]])
				#LUTouts, ClassSum = WNN.predict_1layer(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[k], no_state)
				#LUTouts_class.append(LUTouts)
				_, ClassSum = WNN.predict_1layer(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[k], no_state)
				ClassSum_class.append(ClassSum)
		else:
			# multi-thread
			x_connections = np.array([x[j] for j in input_connections[0]])
			S0, S1, S2, S3, S4 = pool.starmap(WNN.predict_1layer, [(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[0], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[1], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[2], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[3], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[4], no_state) \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[5], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[6], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[7], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[8], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[9], no_state)\
])
			ClassSum_class = [S0[1], S1[1], S2[1], S3[1], S4[1]]

		ClassSum_max = max(ClassSum_class)
		max_idx = [j for j, each in enumerate(ClassSum_class) if each==ClassSum_max]
		if len(max_idx) == 1:
			predicted_label = max_idx[0]
		else:
			predicted_label = random.sample(max_idx,1)[0]
		#predicted_label = np.argmax(ClassSum_class)	

		if predicted_label == y:
			no_correct = no_correct + 1
			no_correct_class[y] = no_correct_class[y] + 1

		#prob = random.uniform(0,1)
		#if train_prob[y] >= prob:
		else:
			# reward
			datain = np.reshape(x_connections, (no_LUTs, no_LUTinputs))
			#TA_states_classes[y] = WNN.reward_all_1layer(TA_states_classes[y], no_state, prob, datain)
			#TA_states_classes[y] = WNN.reward_only0_1layer(TA_states_classes[y], no_state, prob, datain, LUTouts)

			# penalty
			#TA_states_classes[predicted_label] = WNN.penalty_all_1layer(TA_states_classes[predicted_label], no_state, prob, datain)
			#TA_states_classes[predicted_label] = WNN.penalty_only1_1layer(TA_states_classes[predicted_label], no_state, prob, datain, LUTouts_class[predicted_label])

			# reinforcement
			TA_states_classes = WNN.reinforcement_1layer(TA_states_classes, no_state, prob, datain, y, predicted_label)		

		no_trainsamp_class[y] = no_trainsamp_class[y] + 1
		train_acc_class = [float(b/a)*100 for a, b in zip(no_trainsamp_class, no_correct_class)]
		train_prob = [(100-each)/(100-min(train_acc_class)) for each in train_acc_class]
		
		if no_trainsamp%1000 == 0:					
			train_acc = float(no_correct/no_trainsamp)*100
			print('Train accuracy: %.2f%% in %d samples' %(train_acc, no_trainsamp))

			print('Train accuracy per class:')
			print(train_acc_class)

			t_end = time.time()
			comp_time_s = t_end - t_start
			t_start = t_end
			print('Computational time: %.2f s' %comp_time_s)
		#print(ClassSum_class)
		#print(train_prob)
		#print(y)

	# store TA states
	TA_states_classes = np.array(TA_states_classes)
	np.save('TA_states', TA_states_classes)
	#np.savetxt('TA_states.txt', TA_states_classes, fmt='%d')		
	TA_states_classes = TA_states_classes.tolist()		

	# inference
	no_testsamp = len(Y_test)
	no_correct = 0

	no_testsamp_class = [1 for i in range(no_class)]
	no_correct_class = [0 for i in range(no_class)]

	for x, y in zip(X_test, Y_test):
		LUTouts_class = []
		ClassSum_class = []

		if thread == 'single':
			# single thread
			for k in range(no_class):
				x_connections = np.array([x[j] for j in input_connections[k]])
				LUTouts, ClassSum = WNN.predict_1layer(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[k], no_state)
				LUTouts_class.append(LUTouts)
				ClassSum_class.append(ClassSum)
		else:
			# multi-thread
			x_connections = np.array([x[j] for j in input_connections[0]])
			S0, S1, S2, S3, S4 = pool.starmap(WNN.predict_1layer, [(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[0], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[1], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[2], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[3], no_state), \
(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[4], no_state) \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[5], no_state) \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[6], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[7], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[8], no_state), \
#(x_connections, no_LUTinputs, no_LUTs, TA_states_classes[9], no_state)\
])
			ClassSum_class = [S0[1], S1[1], S2[1], S3[1], S4[1]]

		predicted_label = np.argmax(ClassSum_class)

		if predicted_label == y:
			no_correct = no_correct + 1
			no_correct_class[y] = no_correct_class[y] + 1

	no_testsamp_class[y] = no_testsamp_class[y] + 1
	test_acc = float(no_correct/no_testsamp)*100

	no_testsamp_class[y] = no_testsamp_class[y] + 1
	test_acc_class = [float(b/a)*100 for a, b in zip(no_testsamp_class, no_correct_class)]
	print('Test accuracy per class:')
	print(test_acc_class)

	f = open("accuracy_1layer.txt","a")
	f.write("%d\t%.2f%%\t%.2f%%\n" %(i, train_acc, test_acc))
	f.close()
		


