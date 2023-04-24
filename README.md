# Aesthetics-guided Low-Light Image Enhancement

## Introduction
The performance evaluation of low-light image enhancement techniques is highly dependent on subjectivity. Therefore, we argue that embedding human subjective preferences into image enhancement is a key to boost its performance. Unfortunately, currently prevailing methods ignore such a nature but just enumerate a series of potentially valid heuristic constraints for training the enhancement model. This paper presents a new way -- aesthetics-guided low-light image enhancement. It treats low-light image enhancement as a Markov decision process and drives the training through an aesthetics reward. In doing so, each pixel, as an agent, refines its own value by taking a sequential and recursive action, i.e., its corresponding light enhancement curve is estimated separately in each iteration, and then applied to itself to adjust the light. Various experiments demonstrate that introducing the aesthetic assessment indeed leads to better both subjective visual experience and objective evaluation. While our experimental results on various benchmarks also reflect, qualitatively and quantitatively, the advantages of our method over state-of-the-art methods.
## Requirement
* Python 3.5+
* Chainer 5.0+
* Cupy 5.0+
* OpenCV 3.4+
* Torch 1.6

You can install the required libraries by the command `pip install -r requirements.txt`.


## Usage
### Training
If you want to train the model

1. download the LOL dataset or your own dataset
2. 
```
cd Aesthetics-guidedLLE
python train.py
```

### Testing with pretrained models
```
cd Aesthetics-guidedLLE
python test.py
```