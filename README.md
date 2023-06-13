# Image classification using PyTorch 

Change the parameters in the file `conf.yaml`, in case you want, except the outputs parameter.

Run it: `python main.py -c conf/conf.yaml -r [state] -i [image] --preproc [preprocessing method]`

For example: `python main.py -c conf/conf.yaml -r full -i in/SF14/day1_low10.bmp`

For `state` parameter you can choose:

- 'preproc': it only does image preprocessing
- 'prepare': it does Preprocessing -> Train -> Test -> Performance plots and metrics
- 'classify': it classifies the respective image. You must have a trained model first and point to it with the -m flag
- 'full': it does everything from preprocessing to classification

Obs: The results from training and testing are saved in /out/outputs.txt

In conf.yaml we can choose the parameters:

- optim: 'Adam'
- model: {'densenet', 'resnet', 'efficientnet'}
- epochs: Any integer greater than 0
- batch: 16
- lr: !!float 5e-4
- raw: 'in/'
- preproc: {'skip', 'remove_background', 'gaussian', 'median', 'bilateral', 'unsharp'}

