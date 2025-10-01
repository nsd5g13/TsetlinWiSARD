# TsetlinWiSARD: Training of Weightless Neural Networks using Tsetlin Automata 

<!-- ABOUT THE PROJECT -->
## About The Project

TsetlinWiSARD, a Tsetlin Automaton (TA)-based training approch for WiSARD-one of the most extensively studied Weightless Neural Network (WNN) models. WNNs operate on Boolean input features, typically derived by thresholding raw data. We provide source code to booleanize 15 benchmark datasets as preprocessing steps for training. We also provide two existing WiSARD training methods-standard WiSARD and B-bleaching for comparisons.

<!-- GETTING STARTED -->
## Usage

### Booleanization

We provide source code to booleanize 10 TinyML open source datasets:
- [EMG](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
- [Gas sensor array drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
- [Gesture phase segmentation](https://archive.ics.uci.edu/ml/datasets/gesture+phase+segmentation)
- [Human activity](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [Mammographic mass](http://archive.ics.uci.edu/ml/datasets/mammographic+mass)
- [Sensorless drive diagnosis](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis) 
- [Sport activity](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)
- [Statlog (vehicle silhouette)](https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes)
- [Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [Digits](https://scikit-learn.org/1.5/auto_examples/datasets/plot_digits_last_image.html)

We provide source code to booleanize 5 relatively large scaled datasets:
- [MNIST](https://keras.io/api/datasets/mnist/)
- [Kuzushiji MNIST](https://www.tensorflow.org/datasets/catalog/kmnist)
- [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/kmnist)
- [CIFAR10](https://keras.io/api/datasets/cifar10/)
- [Keyword Spotting](https://tensorflow.google.cn/datasets/catalog/speech_commands)

Before Booleanizing, download all raw datasets and put the dataset directory at "/raw_dataset/".
   ```sh
   cd Booleanization
   python3 booleanization.py [dataset_type]
   ```
Options for [dataset_type] are "tinyml" and "large", which produce Booleanized datasets for the above 10 TinyML and the above 5 large scaled datasets, respectively..

### Standard WiSARD training

   ```sh
   cd Training
   ```

   ```sh
   usage: WiSARD.py no_LUTinputs no_LUTs dataset_name thread

 positional arguments:
     no_LUTinputs     Provide the number of inputs per LUT
     no_LUTs	      Provide the number of LUTs per class
     dataset_name    Provide the name of the dataset
     thread		      For single thread training, replace it with "single". Otherwise, training is performed with multi-thread. For multi-thread, properly modify WiSARD.py to ensure each class running per thread.
   ```

Example:
   ```sh
   python3 WiSARD.py 6 300 emg single
   ```

### B-bleaching

   ```sh
   cd Training
   ```

   ```sh
   usage: Bleaching.py no_LUTinputs no_LUTs dataset_name thread

 positional arguments:
     no_LUTinputs     Provide the number of inputs per LUT
     no_LUTs	      Provide the number of LUTs per class
     dataset_name    Provide the name of the dataset
     thread		      For single thread training, replace it with "single". Otherwise, training is performed with multi-thread. For multi-thread, properly modify WiSARD.py to ensure each class running per thread.
   ```

Example:
   ```sh
   python3 Bleaching.py 6 300 emg multi
   ```

### TsetlinWiSARD training

   ```sh
   cd Training
   ```

   ```sh
   usage: TWiSARD.py no_LUTinputs no_LUTs dataset_name epochs thread

 positional arguments:
     no_LUTinputs     Provide the number of inputs per LUT
     no_LUTs	      Provide the number of LUTs per class
     dataset_name    Provide the name of the dataset
     epochs		      Provide the number of training epochs
     thread		      For single thread training, replace it with "single". Otherwise, training is performed with multi-thread. For multi-thread, properly modify WiSARD.py to ensure each class running per thread.
   ```

Example:
   ```sh
   python3 TWiSARD.py 6 300 emg 100 multi
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
