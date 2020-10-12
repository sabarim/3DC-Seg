# 3DC-Seg
This repository contains the official implementation for the paper:

**Making a Case for 3D Convolutions for Object Segmentation in Videos**

*Sabarinath Mahadevan\*, Ali Athar\*,Aljoša Ošep, Laura Leal-Taixé, Bastian Leibe*

BMVC 2020 | [Paper](https://arxiv.org/pdf/2008.11516.pdf) | [Video](https://www.youtube.com/watch?v=vU3g2mpL1XA&ab_channel=RWTHVision) | [Project Page](https://www.vision.rwth-aachen.de/publication/00205/)

## Required Packages

- Python 3.7
- PyTorch 1.4 or greater
- Nvidia-apex: https://github.com/NVIDIA/apex
- tensorboard, pycocotools and other packages listed in requirements.txt

## Setup

1. Clone the repository and append it to the `PYTHONPATH` variable:

   ```bash
   git clone https://github.com/sabarim/3DC-Seg.git
   cd 3DC-Seg
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
2. Create a folder named 'saved_models'

## Checkpoint

1. The trained checkpoint is available in the below given link:


    | Target Dataset        | Datasets Required for Training  | Model Checkpoint |
    |-----------------------| -------------------------------|--------------|
    | DAVIS, FBMS, ViSal | COCO, YouTubeVOS, DAVIS'17 | [link](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/models/bmvc_final.pth)
    
    
## Usage

### Training:

1. Run ```mkdir -p saved_models/csn/```
2. Download the [pretrained backbone weights](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/models/csn.zip) and place it in the folder created above.

```
  python main.py -c run_configs/<name>.yaml --num_workers <number of workers for dataloader> --task train
```

### Inference:

Use the pre-trained checkpoint downloaded from our server along with the provided config files to reproduce the results from Table. 4 and Table. 5 of the paper. Please note that you'll have to use the official [davis evaluation package](https://github.com/davisvideochallenge/davis2017-evaluation) adapted for DAVIS-16 as per the issue listed [here](https://github.com/davisvideochallenge/davis2017-evaluation/issues/4) if you wish to run an evaluation on DAVIS.

1. DAVIS:

```
python main.py -c run_configs/bmvc_final.yaml --task infer --wts <path>/bmvc_final.pth

```

2. DAVIS - Dense

```
python main.py -c run_configs/bmvc_final_dense.yaml --task infer --wts <path>/bmvc_final.pth

```

2. FBMS:

```
python main.py -c run_configs/bmvc_fbms.yaml --task infer --wts <path>/bmvc_final.pth

```

3. ViSal

```
python main.py -c run_configs/bmvc_visal.yaml --task infer --wts <path>/bmvc_final.pth
```


## Pre-computed results

Pre-computed segmentation masks for different datasets can be downloaded from the below given links:

| Target Dataset        | Results  |
|-----------------------|--------------|
| DAVIS | [link](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/results/bmvc_final.zip)
| DAVIS - Dense | [link](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/results/bmvc_final_dense.zip)
| FBMS | [link](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/results/bmvc_fbms.zip)
| ViSal | [link](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/results/bmvc_visal.zip)

