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

# Using the `pegi3s/classify` Docker image

The `pegi3s/classify` Docker image contains these files and all the required dependencies. 

To use it, simply run the following commands, changing `/your/data/dir` with the path to the directory that contains your images.

```
xhost +
docker run --rm -ti -e USERID=$UID -e USER=$USER -e DISPLAY=$DISPLAY -v /var/db:/var/db:Z -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v "/your/data/dir:/data" -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp pegi3s/classify bash -c "conda init bash && cp /data/conf.yaml /conf && python /opt/main.py -c conf.yaml -r full -i /data/Vir_teste.tif"
```
