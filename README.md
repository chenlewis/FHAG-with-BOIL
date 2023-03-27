# FHAG-with-BOIL

## Overview

This is the implementation of the method proposed in "Distortion Model based Spectral Augmentation for Generalized Recaptured Document Detection" with pytorch(1.9.0, gpu version). The associated datasets are available upon request.

## Environment Request

python == 3.10.6
fire == 0.4.0
matplotlib == 3.3.4
numpy == 1.20.2
opencv_python == 4.5.2.54
Pillow== 9.4.0
scikit_learn == 1.2.1
torch == 1.9.0
torchnet == 0.0.4
torchvision == 0.10.0
tqdm == 4.61.1

## Files structure

* data
* models
* utils
* config.py
* main.py
* FHAG.py
* BOIL.py
* analyze.py

## Files Description

```BOIL.py```
Calculate the BOI of existing sample

```FHAG.py```
Augment the recapturing features by amplitude manipulation

## Run

### Train

```terminal
python main.py train 
```

### Test

```terminal
python test.py 
```

### Analyze

```terminal
python analyze.py
```

## Citation

If you use our code please cite: XXX
The datasets presented in our work are available upon request.
