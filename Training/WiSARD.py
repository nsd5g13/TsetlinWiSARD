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

print('# %s dimensions: %d training samples, %d test samples, %d classes, %d Boolean features' %(dataset, len(Y_train), len(Y_test), no_class, no_features))

thread = sys.argv[4]

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
	
f = open("accuracy_wisard.txt","w")

# training
no_trainsamp = 0
no_correct = 0

no_trainsamp_class = [1 for i in range(no_class)]
no_correct_class = [0 for i in range(no_class)]

# training
for x, y in zip(X_train[0:4682], Y_train[0:4682]):
	no_trainsamp = no_trainsamp + 1
	ClassSum_class = []	
		
	# pattern matching
	if thread == 'single':
		# single thread
		for k in range(no_class):
			true_positive = (k == y)
			x_connections = np.array([x[j] for j in input_connections[k]])
			LUTouts_classes[k], ClassSum = WNN.wisard(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], true_positive)
			ClassSum_class.append(ClassSum)
	else:
		# multi-thread
		x_connections = np.array([x[j] for j in input_connections[0]])
		(LUTouts_classes[0], S0), (LUTouts_classes[1], S1), (LUTouts_classes[2], S2), (LUTouts_classes[3], S3), (LUTouts_classes[4], S4), \
(LUTouts_classes[5], S5), (LUTouts_classes[6], S6), (LUTouts_classes[7], S7) = \
pool.starmap(WNN.wisard, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0], (y==0)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], (y==1)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], (y==2)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], (y==3)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], (y==4)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], (y==5)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], (y==6)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], (y==7)) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], (y==8)), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], (y==9))
])
		ClassSum_class = [S0, S1, S2, S3, S4, S5, S6, S7]
	
	ClassSum_max = max(ClassSum_class)
	max_idx = [j for j, each in enumerate(ClassSum_class) if each==ClassSum_max]
	if len(max_idx) == 1:
		predicted_label = max_idx[0]
	else:
		predicted_label = random.sample(max_idx,1)[0]	

	if predicted_label == y:
		no_correct = no_correct + 1
		no_correct_class[y] = no_correct_class[y] + 1
		
	no_trainsamp_class[y] = no_trainsamp_class[y] + 1
	train_acc_class = [float(b/a)*100 for a, b in zip(no_trainsamp_class, no_correct_class)]
		
	#if no_trainsamp%1000 == 0:					
	train_acc = float(no_correct/no_trainsamp)*100
	print('Train accuracy: %.2f%% in %d samples' %(train_acc, no_trainsamp))

	print('Train accuracy per class:')
	print(train_acc_class)

	f.write('%d\t%.2f%%\n' % (no_trainsamp, train_acc))

		#t_end = time.time()
		#comp_time_s = t_end - t_start
		#t_start = t_end
		#print('Computational time: %.2f s' %comp_time_s)
	#print(ClassSum_class)
		#print(train_prob)
	#print(y)	


# inference
no_correct = 0
no_testsamp = 0
no_testsamp_class = [1 for i in range(no_class)]
no_correct_class = [0 for i in range(no_class)]

for x, y in zip(X_test, Y_test):
	no_testsamp = no_testsamp + 1
	ClassSum_class = []	
		
	# pattern matching
	if thread == 'single':
		# single thread
		for k in range(no_class):
			x_connections = np.array([x[j] for j in input_connections[k]])
			_, ClassSum = WNN.wisard(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], False)
			ClassSum_class.append(ClassSum)
	else:
		# multi-thread
		x_connections = np.array([x[j] for j in input_connections[0]])
		(_, S0), (_, S1), (_, S2), (_, S3), (_, S4), \
(_, S5), (_, S6), (_, S7) = \
pool.starmap(WNN.wisard, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0],  False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], False), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], False) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], False), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], False)
])
		ClassSum_class = [S0, S1, S2, S3, S4, S5, S6, S7]
	
	ClassSum_max = max(ClassSum_class)
	max_idx = [j for j, each in enumerate(ClassSum_class) if each==ClassSum_max]
	if len(max_idx) == 1:
		predicted_label = max_idx[0]
	else:
		predicted_label = random.sample(max_idx,1)[0]	

	if predicted_label == y:
		no_correct = no_correct + 1
		no_correct_class[y] = no_correct_class[y] + 1

	no_testsamp_class[y] = no_testsamp_class[y] + 1
	test_acc = float(no_correct/no_testsamp)*100
	print('Test accuracy: %.2f%% in %d samples' %(test_acc, no_testsamp))

	no_testsamp_class[y] = no_testsamp_class[y] + 1
	test_acc_class = [float(b/a)*100 for a, b in zip(no_testsamp_class, no_correct_class)]
	print('Test accuracy per class:')
	print(test_acc_class)

f = open("accuracy_wisard.txt","a")
f.write("%.2f%%\t%.2f%%\n" %(train_acc, test_acc))
f.close()