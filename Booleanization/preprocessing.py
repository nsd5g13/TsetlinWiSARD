import numpy as np
import pathlib
import os, random
#import cv2, librosa
import tensorflow as tf

# -- Encode raw features into one-hot codes -------------------------------------------------
def onehot_encoding(X_raw_features, no_bins):
	thres = []
	for i in range(len(X_raw_features[0])):
		thres.append([])
		for j in range(1, no_bins):
			thres[i].append(np.quantile(X_raw_features[:,i], float(1/no_bins)*j))
	onehot = []

	for i in range(no_bins):
		onehot.append([])
		for j in range(no_bins):
			if i != j:
				onehot[i].append(0)
			else:
				onehot[i].append(1)
	onehot=onehot[::-1]

	X_bool=[]
	for i in range(len(X_raw_features)):
		X_bool.append([])
		for j, each in enumerate(X_raw_features[i]):
			tmp = thres[j]
			tmp = tmp + [each]
			tmp.sort()
			X_index = tmp.index(each)
			if no_bins == 2:
				if X_index == 1:
					X_bool[i].append(1)
				else:
					X_bool[i].append(0)
			else:
				X_bool[i].extend(onehot[X_index])
	
	return X_bool, thres

# -- Encode raw features into thermometer codes -------------------------------------------------
def thermo_encoding(X_raw_features, no_bins):
	thres = []
	for i in range(len(X_raw_features[0])):
		thres.append([])
		for j in range(1, no_bins):
			thres[i].append(np.quantile(X_raw_features[:,i], float(1/no_bins)*j))
	thermo = []

	for i in range(no_bins):
        	thermo.append([])
        	for j in range(no_bins):
                	if i > j:
                        	thermo[i].append(0)
                	else:
                        	thermo[i].append(1)
	thermo=thermo[::-1]

	X_bool=[]
	for i in range(len(X_raw_features)):
		X_bool.append([])
		for j, each in enumerate(X_raw_features[i]):
			tmp = thres[j]
			tmp = tmp + [each]
			tmp.sort()
			X_index = tmp.index(each)
			if no_bins == 2:
				if X_index == 1:
					X_bool[i].append(1)
				else:
					X_bool[i].append(0)
			else:
				X_bool[i].extend(thermo[X_index])
	
	return X_bool

# -- load EMG dataset ---------------------------------------------------------
def EMG_load(dataset_file_path):
	all_subjects = os.walk(dataset_file_path)
	all_dirs = [x[0] for x in all_subjects]
	all_subjects = os.walk(dataset_file_path)
	all_txt = [x[2] for x in all_subjects]
	all_features = []
	all_labels = []
	cnt = 0
	for dir, txt in zip(all_dirs[1:], all_txt[1:]):
		for each in txt:
			file_name = dir + '/' + each
			f = open(file_name, 'r')
			lines = f.readlines()
			f.close()

			print('Read text file: ' + file_name)
			all_features.append([])
			all_labels.append([])
			prev_timepoint = 0
			for each_line in lines[1:]:
				features_label = each_line.split()
				features = [float(x) for x in features_label[:-1]]
				label = features_label[-1]
				curr_timepoint = features[0]
				if curr_timepoint == prev_timepoint + 1:
					all_labels[cnt].append(int(label))
					all_features[cnt].append(features[1:])
				else:	# interpolation
					no_interpolation = int(curr_timepoint - prev_timepoint)
					for i in range(no_interpolation):
						all_labels[cnt].append(int(label))
						all_features[cnt].append(features[1:])
				prev_timepoint = curr_timepoint
			cnt = cnt + 1	

	return all_features, all_labels

# -- relabel unmarked data, if a sample is found in the marked dataset, for EMG dataset
def EMG_relabel(X_Class0, X_notClass0, Y_notClass0):
	Y_relabelled = []
	for each_unmarked in X_Class0:
		FoundOrNot = np.all(X_notClass0==each_unmarked, axis=1)
		matched_indices = np.where(FoundOrNot)
		if np.size(matched_indices, axis=None) != 0:
			new_labels = Y_notClass0[matched_indices]
			new_labels = new_labels.tolist()
			new_label = max(set(new_labels), key=new_labels.count)
			Y_relabelled.append(new_label)
		else:
			Y_relabelled.append(0)
	Y_relabelled = np.array(Y_relabelled)
	return Y_relabelled

# -- load Sports dataset ---------------------------------------------------------
def Sports_load(dataset_file_path):
	all_subjects = os.walk(dataset_file_path)
	all_dirs = [x[0] for x in all_subjects]
	all_dirs.sort(key=len)
	all_dirs = all_dirs[20:]
	all_subjects = os.walk(dataset_file_path)
	all_txt = [x[2] for x in all_subjects]
	all_txt = all_txt[2]
	all_features = []
	all_labels = []
	cnt = 0
	for dir in all_dirs:
		idx1 = dir.index('Sports/a')
		idx2 = dir.index('/p')
		label = dir[idx1 + 8 : idx2]
		label = int(label) - 1
		for txt in all_txt:
			file_name = dir + '/' + txt
			f = open(file_name, 'r')
			lines = f.readlines()
			f.close()

			print('Read text file: ' + file_name)
			all_features.append([])
			all_labels.append([])			
			for each_line in lines:
				features = each_line.replace('\n', '').split(',')
				features = [float(x) for x in features]
				all_features[cnt].append(features)
				all_labels[cnt].append(label)
			cnt = cnt + 1	

	return all_features, all_labels

# -- load Human Activity dataset ---------------------------------------------------------
def HAR_load(dataset_file_path):
	TrainOrTest = dataset_file_path.split('/')[2]
	file_name = dataset_file_path + '/X_' + TrainOrTest + '.txt'
	f = open(file_name ,'r')
	X_lines = f.readlines()
	f.close()

	file_name = dataset_file_path + '/y_' + TrainOrTest + '.txt'
	f = open(file_name ,'r')
	y_lines = f.readlines()
	f.close()		

	#utilized_feature_indices = [0, 1, 2, 9, 10, 11, 40, 41, 42]
	utilized_feature_indices = [i for i in range(560)]

	all_features = []
	all_labels = []
	for X_line, y_line in zip(X_lines, y_lines):
		features = X_line.replace('\n', '').split()
		features = [features[i] for i in utilized_feature_indices]
		features = [float(x) for x in features]
		all_features.append(features)
		label = y_line.replace('\n', '')
		label = int(label) - 1
		all_labels.append(label)	

	return all_features, all_labels

# -- load Gesture Phase Segmentation dataset ---------------------------------------------------------
def Gesture_load(dataset_file_path):
	labels = ['Rest', 'Preparation', 'Stroke', 'Hold', 'Retraction']
	#labels = ['D', 'P', 'S', 'H', 'R']
	all_subjects = os.walk(dataset_file_path)
	all_csv = [x[2] for x in all_subjects][0]
	all_raw_csv = []
	all_features = []
	all_labels = []
	for each in all_csv:
		if 'raw' in each:
			all_raw_csv.append(each)
	for each in all_raw_csv:
		file_name = dataset_file_path + '/' + each
		f = open(file_name, 'r')
		lines = f.readlines()
		f.close()

		for each_line in lines[1:]:
			features_label = each_line.replace('\n', '').split(',')
			features = [float(x) for x in features_label[:-2]]
			label = features_label[-1]
			label = labels.index(label)
			all_features.append(features)
			all_labels.append(label)

	return all_features, all_labels

# -- load Gas Sensor Array Drift dataset ---------------------------------------------------------
def Gas_load(dataset_file_path):
	all_subjects = os.walk(dataset_file_path)
	all_bat = [x[2] for x in all_subjects][0]
	all_features = []
	all_labels = []
	for each in all_bat:
		file_name = dataset_file_path + '/' + each
		f = open(file_name, 'r')
		lines = f.readlines()
		f.close()

		for each_line in lines[1:]:
			features_label = each_line.replace('\n', '').split()
			label = int(features_label[0])-1
			all_labels.append(label)
			features = [float(x.split(":")[1]) for x in features_label[1:]]
			all_features.append(features)

	return all_features, all_labels

# -- load Statlog Vehicle Silhouettes dataset ---------------------------------------------------------
def Statlog_load(dataset_file_path):
	labels = ['van', 'saab', 'bus', 'opel']
	all_subjects = os.walk(dataset_file_path)
	all_bat = [x[2] for x in all_subjects][0]
	all_features = []
	all_labels = []
	for each in all_bat[2:]:
		file_name = dataset_file_path + '/' + each
		f = open(file_name, 'r')
		lines = f.readlines()
		f.close()

		for each_line in lines:
			features_label = each_line.replace('\n', '').split()
			label = labels.index(features_label[-1])
			all_labels.append(label)
			features = [int(x) for x in features_label[:-1]]
			all_features.append(features)

	return all_features, all_labels

# -- load Mammographic Mass dataset ---------------------------------------------------------
def Mammography_load(dataset_file_path):
	f = open(dataset_file_path, 'r')
	lines = f.readlines()
	f.close()

	all_features = []
	all_labels = []
	for each_line in lines:
		features_label = each_line.replace('\n', '').split(',')
		label = int(features_label[-1])
		all_labels.append(label)
		all_features.append(features_label[:-1])
	return all_features, all_labels

# -- onehot/thermometer encoding for Mammographic Mass dataset -----------------------------------------------
def Mammography_encoding(all_features, no_bins, coding):
	no_features = len(all_features[0])
	unmissing_features = [[] for i in range(no_features)]
	for features in all_features:
		for i in range(no_features):
			if features[i] != '?':
				unmissing_features[i].append(int(features[i]))

	thres = [[] for i in range(no_features)]
	for i in range(no_features):
		for j in range(1, no_bins):
			thres[i].append(np.quantile(unmissing_features[i], float(1/no_bins)*j))

	encoded = []
	for i in range(no_bins):
		encoded.append([])
		for j in range(no_bins):
			if coding == 'onehot':
				if i != j:
					encoded[i].append(0)
				else:
					encoded[i].append(1)
			elif coding == 'thermometer':
				if i > j:
                        		encoded[i].append(0)
				else:
                        		encoded[i].append(1)
			else:
				print('The given coding method %s is not recognized' %coding)
	encoded=encoded[::-1]

	X_bool = []
	for features in all_features:
		bool_features = []
		for i, feature in enumerate(features):
			if feature != '?':
				feature = int(feature)
				tmp = thres[i]
				tmp = tmp + [feature]
				tmp.sort()
				X_index = tmp.index(feature)
			else:
				random.seed(1)
				X_index = random.randint(0, no_bins)
			if no_bins == 2:
				if X_index == 1:
					bool_features.append(1)
				else:
					bool_features.append(0)
			else:
				bool_features.extend(encoded[X_index])
		X_bool.append(bool_features)
	return X_bool

# -- load Sensorless Drive Diagnosis dataset ---------------------------------------------------------
def Sensorless_load(dataset_file_path):
	f = open(dataset_file_path, 'r')
	lines = f.readlines()
	f.close()

	all_features = []
	all_labels = []
	for each_line in lines:
		features_label = each_line.replace('\n', '').split()
		label = int(features_label[-1]) - 1
		all_labels.append(label)
		features = [float(x) for x in features_label[:-1]]
		all_features.append(features)
	return all_features, all_labels				
	
					
# -- Perform certain pooling method for each given number of timesteps --------------------------
def pooling(all_features, all_labels, winLength, poolMethod):
	X = []
	Y = []
	for subject_features, subject_labels in zip(all_features, all_labels):
		subject_X = []
		subject_Y = []

		no_windows = int(np.floor((len(subject_features)-winLength)/winLength))
		start_time = 0
		end_time = start_time + winLength
		for window in range(no_windows):
			subject_window = subject_features[start_time:end_time]
			subject_window = np.array(subject_window)
			if poolMethod == 'max':
				pool_values = subject_window.max(axis=0)
			elif poolMethod == 'avg':
				pool_values = np.mean(subject_window, axis=0)
			elif poolMethod == 'rms':		# root mean square
				tmp = np.square(subject_window)
				tmp = np.sum(tmp, axis=0)
				pool_values = np.sqrt(tmp)
			elif poolMethod == 'mav':		# mean absolute value
				tmp = np.abs(subject_window)
				pool_values = np.mean(tmp, axis=0)				
			else:
				print('The specified pooling method is not recognized!')

			window_labels = subject_labels[start_time:end_time]
			window_label = max(set(window_labels), key=window_labels.count)
			
			subject_Y.append(window_label)
			subject_X.append(pool_values.tolist())			

			start_time = start_time + winLength
			end_time = end_time + winLength	

		X.append(subject_X)
		Y.append(subject_Y)
	return X, Y

# -- Generate features based on given number of timesteps (window length) and stride --------------------------
def SlidingWindow(all_features, all_labels, window_length, stride):
	X = []
	Y = []
	cnt = 0
	for subject_features, subject_labels in zip(all_features, all_labels):
		no_windows = int(np.floor((len(subject_labels)-window_length)/stride) + 1)
		start_time = 0
		end_time = start_time + window_length
		for window in range(no_windows):
			X.append([])
			for i in range(window_length):
				X[cnt].extend(subject_features[start_time+i])

			window_labels = subject_labels[start_time:end_time]
			window_label = max(set(window_labels), key=window_labels.count)
			Y.append(window_label)
			
			start_time = start_time + stride
			end_time = start_time + window_length
			cnt = cnt + 1
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

# -- Split dataset into training/test set
def DatasetSplit(X, Y, TrainingSetRatio):
	random.seed(1)
	n = random.sample(range(len(Y)), len(Y))
			
	X_train = []
	Y_train = []
	for each in n[0:int(len(Y)*TrainingSetRatio)]:
		X_train.append(X[each])
		Y_train.append(Y[each])

	X_test = []
	Y_test = []
	for each in n[int(len(Y)*TrainingSetRatio):]:
		X_test.append(X[each])
		Y_test.append(Y[each])

	X_train=np.array(X_train)
	Y_train=np.array(Y_train)
	X_test=np.array(X_test)
	Y_test=np.array(Y_test)

	return X_train, Y_train, X_test, Y_test

# -- preprocessing, specifically for CIFAR ---------------------------------------------------------
def CIFAR_HOG(X_raw_features):
	patch_size = 0

	imageSize = 32  #The size of the original image - in pixels - assuming this is a square image
	channels = 3    #The number of channels of the image. A RBG color image, has 3 channels

	winSize = imageSize
	blockSize = 24
	blockStride = 8
	cellSize = 8
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = True
	nlevels = 64
	signedGradient = True
	hog = cv2.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride,blockStride),(cellSize,cellSize),nbins,derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)

	fd = hog.compute(X_raw_features[0])
	fds = []
	X_bool = np.empty((X_raw_features.shape[0], fd.shape[0]), dtype=np.uint32)
	for i in range(X_raw_features.shape[0]):
		fd = hog.compute(X_raw_features[i])
		fds.append(fd)

	fds = np.array(fds)
	fds = np.transpose(fds)
	fd_quantiles = []
	for fd in fds:
		fd_quantiles.append(np.quantile(fd, 0.5))

	for i in range(X_raw_features.shape[0]):
		fd = hog.compute(X_raw_features[i])
		for j, each_fd in enumerate(fd):
			if each_fd >= fd_quantiles[j]:
				X_bool[i][j] = 1
			else:
				X_bool[i][j] = 0

	return X_bool

# -- preprocessing, specifically for KWS ---------------------------------------------------------
# refer: https://www.tensorflow.org/tutorials/audio/simple_audio
# ------------------------------------------------------------------------------------------------
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
	# Convert the waveform to a spectrogram via a STFT.
	stfts = tf.signal.stft(waveform, frame_length=2560, frame_step=480)
	# Obtain the magnitude of the STFT.
	spectrograms = tf.abs(stfts)
	# Add a `channels` dimension, so that the spectrogram can be used
	# as image-like input data with convolution layers (which expect
	# shape (`batch_size`, `height`, `width`, `channels`).
	spectrograms = spectrograms[..., tf.newaxis]
	return spectrograms

def make_spec_ds(ds):
  return ds.map(map_func=lambda audio,label: (get_spectrogram(audio), label),num_parallel_calls=tf.data.AUTOTUNE)

def kws_dataset(dataset_path):
	data_dir = pathlib.Path(dataset_path)
	if not data_dir.exists():
		# properly change the origin
		tf.keras.utils.get_file(origin='file:///mnt/c/Users/nsd221/Documents/TsetlinWiSARD/Booleanization/raw_dataset/mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir='data')
	train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(directory=data_dir, batch_size=1, validation_split=0.2, seed=0, output_sequence_length=16000, subset='both', sampling_rate=16000)
	label_names = np.array(train_ds.class_names)
	print("KWS label names:", label_names)

	train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
	val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

	train_spectrogram_ds = make_spec_ds(train_ds)
	train_spectrogram_ds = train_spectrogram_ds.unbatch()
	all_train_x = []
	all_train_y = []
	for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(6400):
		train_x = example_spectrograms.numpy()
		mels = librosa.feature.melspectrogram(S=train_x, sr=8000)
		log_mels = librosa.core.amplitude_to_db(mels, ref=np.max)
		mfcc = librosa.feature.mfcc(S=log_mels, sr=8000, n_mfcc=13)	
		train_x = mfcc.flatten()
		all_train_x.append(train_x.tolist())
		train_y = example_spect_labels.numpy()
		train_y = train_y.flatten()
		all_train_y.extend(train_y.tolist())
	train_x = np.array(all_train_x)
	train_y = np.array(all_train_y)

	val_spectrogram_ds = make_spec_ds(val_ds)
	val_spectrogram_ds = val_spectrogram_ds.unbatch()
	all_test_x = []
	all_test_y = []
	for example_spectrograms, example_spect_labels in val_spectrogram_ds.take(1600):
		test_x = example_spectrograms.numpy()
		mels = librosa.feature.melspectrogram(S=test_x, sr=8000)
		log_mels = librosa.core.amplitude_to_db(mels, ref=np.max)
		mfcc = librosa.feature.mfcc(S=log_mels, sr=8000, n_mfcc=13)
		test_x = mfcc.flatten()
		all_test_x.append(test_x.tolist())
		test_y = example_spect_labels.numpy()
		test_y = test_y.flatten()
		all_test_y.extend(test_y.tolist())
	test_x = np.array(all_test_x)
	test_y = np.array(all_test_y)

	return train_x, train_y, test_x, test_y