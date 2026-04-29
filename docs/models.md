# Models

Three models are implemented, each with a different approach to represent jet data. For each model, four Python files are provided: `dataset.py`, `model.py`, `train.py`, and `evaluate.py`.

## BNN

Bayesian Neural Network (PyTorch). Flattened input (80 constituents × 7 features) through 5 fully connected layers with batch normalization, ReLU, and dropout. Follows the DNN architecture from Appendix A of *Constituent-Based Top-Quark Tagging with the ATLAS Detector* (2022). Uses Monte Carlo dropout at inference for uncertainty estimation.

## ResNet50

Residual network (PyTorch). Jet constituents are binned into 64×64 pT-weighted images in eta-phi space. Uses a Bottleneck-based ResNet50 with layers [3, 4, 6, 3] starting from 16 initial planes.

## ParticleNet

Graph neural network (Keras/TensorFlow). Operates on the point cloud of constituents using k-nearest neighbor graphs (k=18) with 3 EdgeConv blocks, following the [official implementation](https://github.com/hqucms/ParticleNet/blob/master/tf-keras/tf_keras_model.py).

## Training details

All models are trained with the Adam optimizer and binary cross-entropy loss, with a 90/10 train/validation split. The best checkpoint (lowest validation loss) is saved to `checkpoints/<model>/best_model.pt` and training curves are written to `figures/`.

For the BNN, evaluation uses Monte Carlo dropout with 10 stochastic forward passes to produce mean predictions and uncertainty estimates.

Metrics are computed at two signal efficiency working points (TPR=0.5 and TPR=0.8): accuracy, AUC, recall, precision, TPR, FPR, and background rejection (1/FPR).
