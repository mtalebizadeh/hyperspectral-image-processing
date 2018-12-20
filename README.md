# Deep learning for hyperspectral image processing
This repository provides three Jupyter notebooks streamlining the application of deep learning algorithms for
hyperspectral image classification. The three classification methods are based on Multi-Layer
Perceptron [(MLP)](deep_learning_MLP.ipynb), 2-Dimensional Convolutional Neural Network [(2-D CNN)](deep_learning_2D_CNN.ipynb),
and 3-D Convolutional Neural Network models [(3-D CNN)](deep_learning_3D_CNN.ipynb). The different stages of the analysis
including pre-processing of raw data, sampling of train and test datasets, dimensionality reduction,
data rescaling, model training and prediction, and a set of result visualization tools are made possible through
the ['img_util'](img_util.py) module.  
 
 
## Package dependency
The notebooks were tested in an environment using python 3.6.6, numpy 1.15.2, keras 2.2.2, scipy 1.1.0, matplotlib 2.2.3.            


## Data source
The Indian Pine hyperspectral dataset and its corresponding ground truth data can be downloaded from the following website:
http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
