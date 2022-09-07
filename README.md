# Spatial wise dynamic neural network based on RDUNet configuration

## Dependencies
- Python 3.6
- PyTorch 1.5.1
- pytorch-msssim 0.2.0
- ptflops 0.6.3
- tqdm 4.48.2
- scikit-image 0.17.2
- yaml 0.2.5

## Dataset
For training: [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) dataset. You need to [download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) the dataset for training the model and put the high-resolution image folders in the './Dataset' folder. You can modify the ```train_files.txt``` and ```val_files.txt``` to load only part of the dataset.

## Training
Default parameters used in the paper are set in the ```config.yaml``` file:

```
patch size: 64
batch size: 16
learning rate: 1.e-4
weight decay: 1.e-5
scheduler gamma: 0.5
scheduler step: 3
epochs: 21
```

To train the model use the following command:

```python main_train.py```

## Test

```python main_test.py```

## Results


