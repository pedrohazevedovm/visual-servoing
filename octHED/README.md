## A Octave implementatio of HED

#### Introduction

Dataset Drive: https://drive.google.com/drive/folders/1VcO4dnEVRsSBdTxBN0itiLKNQ_kUbxsv?usp=drive_link

Source implementation: https://github.com/xwjabc/hed.git

This is a PyTorch reimplementation of [Holistically-nested Edge Detection (HED)](https://arxiv.org/abs/1504.06375).The dependencies are specified at the poetry file.   

#### Instructions

##### Prepare

1. Download and extract the dataset:

   ```bash
   cd hed
   wget https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
   tar xvf ./hed-data.tar
   ```

##### Train and Evaluate

1. Train:

   ```bash
   python3 main.py [parameters]
   ```
   The code has 
   The results are in `output` folder. In the default settings, the HED model is trained for 40 epochs.

##### Parameters List

* `--test`: Flag to execute only the test/inference procedure (skips training).
* `--graph`: Flag to save the loss curve plot (`graph.jpg`) in the output folder.
* `--save_parameters`: Flag to save a `YAML` file containing the training configurations and metrics.
* `--train_batch_size`: Batch size used for training (Default: 1).
* `--test_batch_size`: Batch size used for testing (Default: 1).
* `--train_iter_size`: Number of batches for gradient accumulation before performing a weight update (Default: 10).
* `--max_epoch`: Maximum number of training epochs (Default: 40).
* `--print_freq`: Frequency (in number of batches) to display logs and save intermediate images (Default: 500).
* `--lr`: Initial learning rate (Default: 1e-6).
* `--lr_stepsize`: Step interval (in iterations) for the learning rate decay (Default: 1e4).
* `--lr_gamma`: Decay factor (gamma) applied to the learning rate (Default: 0.1).
* `--momentum`: Momentum factor for the SGD optimizer (Default: 0.9).
* `--weight_decay`: Weight decay penalty / L2 regularization (Default: 2e-4).
* `--loss`: Loss function to be used, such as `weight_cross_entropy` or `ranked_loss` (Default: 'weight_cross_entropy').
* `--only_flops`: Flag to only calculate and display the model operations (FLOPs/MACs) and then exit.
* `--fine_tuning`: Flag to enable weight initialization based on ImageNet VGG.
* `--vgg16_caffe`: Path to the original Caffe VGG-16 parameters file.
* `--checkpoint`: Path to a PyTorch checkpoint (`.pt`) to resume training or run tests.
* `--caffe_model`: Path to a pre-trained HED model in the original Caffe format.
* `--output`: Target directory for logs, checkpoints, and generated images (Default: './output').
* `--dataset_folder`: Path to the dataset folder (Default: './data/HED-BSDS').
* `--dataset_name`: Name of the loaded dataset, e.g., BSDS or BIPED (Default: 'BSDS').
* `--cpu`: Flag that forces the model to run on the CPU, ignoring the CUDA GPU.
* `--model`: Architecture name of the selected model, e.g., HED or OCTHED (Default: 'HED').
* `--alpha`: Alpha parameter for channel splitting, used only with the OCTHED model (Default: -1).
* `--octave_layers`: List of layers that will utilize Octave Convolutions, when applicable (e.g., conv2 conv3).
* `--HSV`: Flag to convert the dataset images color space from RGB to HSV.

##### Model Names
Use a factory approach to make easier to handle with experiments, at `models/model_factory.py`.

* `'HED'`: Baseline model that does not utilize any octave operations.

* `'OCTHED'`: Model featuring configurable octave layers. Applies alpha_in = 0 and alpha_out = 0 to each convolution block, which generally reduces computation speed.

* `'OCTHEDFULL'`': Model where side outputs utilize octave convolutions. Preserves the high-low frequency branches throughout the entire VGG backbone (except for the conv1 block).

* `'EXCITHED'`': Model that integrates Channel Excitation Attention at the end of each convolutional block.

* `'EXCITOCTHED'`': Model combining Excitation Attention at the end of each convolutional block with octave convolutions, following the same implementation as `'OCTHED`'.

* `'ENCODER_DECODER'`': Architecture inspired by UAED that combines VGG and U-Net. Requires a different dataset, which is included in the shared drive.

* `'OCTENCODER_DECODER'`': Architecture inspired by UAED that combines VGG and Octave-U-Net. Requires a different dataset, which is included in the shared drive.

* `'FFCHED'`': Model incorporating Fast Fourier Convolutions (FFC). Currently delivers sub-optimal results and is open to improvements.

* `'OCTHEDFULL_SIDE_EXTRA'`': Model featuring an additional layer to "undo" the octave operation and generate the final score. Currently underperforming, as weights tend to output gray images (open to improvements).

* `'DENSEHED'`': Model utilizing dense skip connections. Shows the poorest performance in terms of both execution time and evaluation metrics.


##### Evaluate my pretrained model

1. Evaluate the pre-trained version:

   ```bash
   python main.py --checkpoint ./checkpoint/epoch-9-checkpoint.pt --output test_octave --model OCTHEDFULL --alpha 0.5 --test
   ```
