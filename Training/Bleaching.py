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
	
f = open("accuracy_bleaching.txt","w")

# training

no_correct = 0
no_trainsamp = 0
no_trainsamp_class = [1 for i in range(no_class)]
no_correct_class = [0 for i in range(no_class)]

# training
for x, y in zip(X_train, Y_train):
	no_trainsamp = no_trainsamp + 1
	ClassSum_class = []	
		
	# pattern matching
	if thread == 'single':
		# single thread
		for k in range(no_class):
			true_positive = (k == y)
			x_connections = np.array([x[j] for j in input_connections[k]])
			LUTouts_classes[k] = WNN.bleaching(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], true_positive)
	else:
		# multi-thread
		x_connections = np.array([x[j] for j in input_connections[0]])
		LUTouts_classes[0], LUTouts_classes[1], LUTouts_classes[2], LUTouts_classes[3], LUTouts_classes[4], LUTouts_classes[5] = \
pool.starmap(WNN.bleaching, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0], (y==0)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], (y==1)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], (y==2)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], (y==3)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], (y==4)), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], (y==5)) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], (y==6)), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], (y==7)), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], (y==8)), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], (y==9))
])

	print('LUT output increments based on %d samples' %no_trainsamp)	

# threshold determination
no_trainsamp = 0
all_b = []
for x, y in zip(X_train[0:1000], Y_train[0:1000]):
	no_trainsamp = no_trainsamp + 1
	ClassSum_classes = [0 for i in range(no_class)]
	prev_ClassSum_classes = [0 for i in range(no_class)]
	b = 0
	while max(ClassSum_classes) == max(prev_ClassSum_classes):
		if thread == 'single':
			# single thread
			for k in range(no_class):
				x_connections = np.array([x[j] for j in input_connections[k]])
				prev_ClassSum_classes[k] = WNN.bleaching_response(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], b)
				ClassSum_classes[k] = WNN.bleaching_response(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], b+1)
		else:
			# multi-thread
			x_connections = np.array([x[j] for j in input_connections[0]])
			prev_ClassSum_classes[0], prev_ClassSum_classes[1], prev_ClassSum_classes[2], prev_ClassSum_classes[3], prev_ClassSum_classes[4], \
prev_ClassSum_classes[5] = \
pool.starmap(WNN.bleaching_response, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], b) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], b)
])
			ClassSum_classes[0], ClassSum_classes[1], ClassSum_classes[2], ClassSum_classes[3], ClassSum_classes[4], ClassSum_classes[5] = \
pool.starmap(WNN.bleaching_response, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0], b+1), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], b+1), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], b+1), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], b+1), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], b+1), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], b+1) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], b+1), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], b+1), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], b+1), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], b+1)
])

		b = b + 1
	all_b.append(b-1)
	print('Bleaching threshold is determined based on %d samples' %no_trainsamp)
all_b = list(set(all_b))
all_b_acc = []
cnt = 0
for b in all_b:
	no_trainsamp = 0
	no_correct = 0
	cnt = cnt + 1

	no_trainsamp_class = [1 for i in range(no_class)]
	no_correct_class = [0 for i in range(no_class)]
	for x, y in zip(X_train[0:1000], Y_train[0:1000]):
		ClassSum_classes = [0 for i in range(no_class)]
		no_trainsamp = no_trainsamp + 1
		if thread == 'single':
			# single thread
			for k in range(no_class):
				x_connections = np.array([x[j] for j in input_connections[k]])
				ClassSum_classes[k] = WNN.bleaching_response(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], b)
		else:
			# multi-thread
			x_connections = np.array([x[j] for j in input_connections[0]])
			ClassSum_classes[0], ClassSum_classes[1], ClassSum_classes[2], ClassSum_classes[3], ClassSum_classes[4], ClassSum_classes[5] = \
pool.starmap(WNN.bleaching_response, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], b) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], b)
])

		ClassSum_max = max(ClassSum_classes)
		max_idx = [j for j, each in enumerate(ClassSum_classes) if each==ClassSum_max]
		if len(max_idx) == 1:
			predicted_label = max_idx[0]
		else:
			predicted_label = random.sample(max_idx,1)[0]	

		if predicted_label == y:
			no_correct = no_correct + 1
			no_correct_class[y] = no_correct_class[y] + 1
		
		no_trainsamp_class[y] = no_trainsamp_class[y] + 1
		train_acc_class = [float(b/a)*100 for a, b in zip(no_trainsamp_class, no_correct_class)]
							
		train_acc = float(no_correct/no_trainsamp)*100
		print('Train accuracy: %.2f%% in %d samples, b=%d, %d/%d' %(train_acc, no_trainsamp, b, cnt, len(all_b)))

	print('Train accuracy per class:')
	print(train_acc_class)
	all_b_acc.append(train_acc)

final_b = all_b[np.argmax(all_b_acc)]
	
# inference
no_correct = 0
no_testsamp = 0
no_testsamp_class = [1 for i in range(no_class)]
no_correct_class = [0 for i in range(no_class)]

for x, y in zip(X_test, Y_test):
	no_testsamp = no_testsamp + 1
	ClassSum_classes = [0 for i in range(no_class)]	
	if thread == 'single':
		# single thread
		for k in range(no_class):
			x_connections = np.array([x[j] for j in input_connections[k]])
			ClassSum_classes[k] = WNN.bleaching_response(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[k], final_b)
	else:
		# multi-thread
		x_connections = np.array([x[j] for j in input_connections[0]])
		ClassSum_classes[0], ClassSum_classes[1], ClassSum_classes[2], ClassSum_classes[3], ClassSum_classes[4], ClassSum_classes[5] = \
pool.starmap(WNN.bleaching_response, [(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[0], final_b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[1], final_b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[2], final_b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[3], final_b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[4], final_b), \
(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[5], final_b) \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[6], final_b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[7], final_b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[8], final_b), \
#(x_connections, no_LUTinputs, no_LUTs, LUTouts_classes[9], final_b)
])
	
	ClassSum_max = max(ClassSum_classes)
	max_idx = [j for j, each in enumerate(ClassSum_classes) if each==ClassSum_max]
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

f = open("accuracy_bleaching.txt","a")
f.write("Train accuracy:\n")
for acc, b in zip(all_b_acc, all_b):
	f.write("%d\t%.2f%%\n" %(b, acc))
f.write("Test accuracy:\n%d\t%.2f%%\n" %(final_b, test_acc))
f.close()