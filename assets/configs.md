# Training configurations

This document provides guidelines for selecting appropriate training options for various scenarios, as well as an extensive list of recommended configurations.

#### Example

In the remainder of this document, we summarize each configuration as follows:

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :--------------: | :--------------: | :------------: | :--
| <sub>CREPS</sub> | <sub>18.47</sub> | <sub>12.29</sub> | <sub>4.3</sub> | <sub>`--cfg=creps --gpus=8 --batch=32 --gamma=8.2 --mirror=1`</sub>

This corresponds to the following command line:

```.bash
# Train CREPS for AFHQv2 using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=creps --data=~/datasets/afhqv2-512x512.zip \
  --gpus=8 --batch=32 --gamma=8.2 --mirror=1
```

Explanation of the columns:
- **Config**: CREPS, StyleGAN3-T (translation equiv.), StyleGAN3-R (translation and rotation equiv.), or StyleGAN2. Reflects the value of `--cfg`.
- **s/kimg**: Raw training speed, measured separately on Tesla V100 and A100 using our recommended Docker image. The number indicates how many seconds, on average, it takes to process 1000 images from the training set. The number tends to vary slightly over the course of training; typically by no more than &plusmn;20%.
- **GPU mem**: Maximum GPU memory usage observed during training, reported in gigabytes per GPU. The above example uses 8 GPUs, which means that the total GPU memory usage is around 34.4 GB.
- **Options**: Command line options for `train.py`, excluding `--outdir` and `--data`.

#### Total training time

In addition the raw s/kimg number, the training time also depends on the `--kimg` and `--metric` options. `--kimg` controls the total number of training iterations and is set to 25000 by default. This is long enough to reach convergence in typical cases, but in practice the results should already look quite reasonable around 5000 kimg. `--metrics` determines which quality metrics are computed periodically during training. The default is `fid50k_full`, which increases the training time slightly; typically by no more than 5%. The automatic computation can be disabled by specifying `--metrics=none`.

In the above example, the total training time on V100 is approximately 18.47 s/kimg * 25000 kimg * 1.05 &thickapprox; 485,000 seconds &thickapprox; 5 days and 14 hours. Disabling metric computation (`--metrics=none`) reduces this to approximately 5 days and 8 hours.

## General guidelines

The most important hyperparameter that needs to be tuned on a per-dataset basis is the R<sub>1</sub> regularization weight, `--gamma`, that must be specified explicitly for `train.py`. As a rule of thumb, the value of `--gamma` scales quadratically with respect to the training set resolution: doubling the resolution (e.g., 256x256 &rarr; 512x512) means that `--gamma` should be multiplied by 4 (e.g., 2 &rarr; 8). The optimal value is usually the same for `--cfg=creps`, `--cfg=stylegan3-t` and `--cfg=stylegan3-r`, but considerably lower for `--cfg=stylegan2`.

In practice, we recommend selecting the value of `--gamma` as follows:
- Find the closest match for your specific case in this document (config, resolution, and GPU count).
- Try training with the same `--gamma` first.
- Then, try increasing the value by 2x and 4x, and also decreasing it by 2x and 4x.
- Pick the value that yields the lowest FID.

The results may also be improved by adjusting `--mirror` and `--aug`, depending on the training data. Specifying `--mirror=1` augments the dataset with random *x*-flips, which effectively doubles the number of images. This is generally beneficial with datasets that are horizontally symmetric (e.g., FFHQ), but it can be harmful if the images contain noticeable asymmetric features (e.g., text or letters). Specifying `--aug=noaug` disables adaptive discriminator augmentation (ADA), which may improve the results slightly if the training set is large enough (at least 100k images when accounting for *x*-flips). With small datasets (less than 30k images), it is generally a good idea to leave the augmentations enabled.

It is possible to speed up the training by decreasing network capacity, i.e., `--cbase=16384`. This typically leads to lower quality results, but the difference is less pronounced with low-resolution datasets (e.g., 256x256).

#### Scaling to different number of GPUs

You can select the number of GPUs by changing the value of `--gpu`; this does not affect the convergence curves or training dynamics in any way. By default, the total batch size (`--batch`) is divided evenly among the GPUs, which means that decreasing the number of GPUs yields higher per-GPU memory usage. To avoid running out of memory, you can decrease the per-GPU batch size by specifying `--batch-gpu`, which performs the same computation in multiple passes using gradient accumulation.

By default, `train.py` exports network snapshots once every 200 kimg, i.e., the product of `--snap=50` and `--tick=4`. When using few GPUs (e.g., 1&ndash;2), this means that it may take a very long time for the first snapshot to appear. We recommend increasing the snapshot frequency in such cases by specifying `--snap=20`, `--snap=10`, or `--snap=5`.

Note that the configurations listed in this document have been specifically tuned for 8 GPUs. The safest way to scale them to different GPU counts is to adjust `--gpu`, `--batch-gpu`, and `--snap` as described above, but it may be possible to reach faster convergence by adjusting some of the other hyperparameters as well. Note, however, that adjusting the total batch size (`--batch`) requires some experimentation; decreasing `--batch` usually necessitates increasing regularization (`--gamma`) and/or decreasing the learning rates (most importantly `--dlr`).

#### Transfer learning

Transfer learning makes it possible to reach very good results very quickly, especially when the training set is small and/or the images resemble the ones produced by a pre-trained model. To enable transfer learning, you can point `--resume` to one of the pre-trained models in a correct path. For example:

```.bash
# Fine-tune CREPS for MetFaces using 1 GPU, starting from the pre-trained FFHQ pickle.
python train.py --outdir=~/training-runs --cfg=creps --data=~/datasets/metfaces-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=creps-ffhq-512x512.pkl
```

The pre-trained model should be selected to match the specified config, resolution, and architecture-related hyperparameters (e.g., `--cbase`, `--map-depth`, and `--mbstd-group`). You check this by looking at the `fakes_init.png` exported by `train.py` at the beginning; if the configuration is correct, the images should look reasonable.

With transfer learning, the results may be improved slightly by adjusting `--freezed`, in addition to the above guidelines for `--gamma`, `--mirror`, and `--aug`. In our experience, `--freezed=10` and `--freezed=13` tend to work reasonably well.

## Recommended configurations

This section lists recommended settings for StyleGAN3-T and StyleGAN3-R for different resolutions and GPU counts, selected according to the above guidelines. These are intended to provide a good starting point when experimenting with a new dataset. Please note that many of the options (e.g., `--gamma`, `--mirror`, and `--aug`) are still worth adjusting on a case-by-case basis.

#### 128x128 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :--------------: | :--------------: | :------------: | :--
| <sub>CREPS</sub> | <sub>1</sub> | <sub>73.68</sub> | <sub>27.20</sub> | <sub>7.2</sub> | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10`</sub>
| <sub>CREPS</sub> | <sub>2</sub> | <sub>37.30</sub> | <sub>13.74</sub> | <sub>7.1</sub> | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=0.5 --snap=20`</sub>
| <sub>CREPS</sub> | <sub>4</sub> | <sub>20.66</sub> | <sub>7.52</sub>  | <sub>4.1</sub> | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=0.5`</sub>
| <sub>CREPS</sub> | <sub>8</sub> | <sub>11.31</sub> | <sub>4.40</sub>  | <sub>2.6</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=0.5`</sub>

#### 256x256 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :--------------: | :--------------: | :------------: | :--
| <sub>CREPS</sub> | <sub>1</sub> | <sub>89.15</sub> | <sub>49.81</sub> | <sub>9.5</sub> | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=2 --batch-gpu=16 --snap=10`</sub>
| <sub>CREPS</sub> | <sub>2</sub> | <sub>45.45</sub> | <sub>25.05</sub> | <sub>9.3</sub> | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=2 --snap=20`</sub>
| <sub>CREPS</sub> | <sub>4</sub> | <sub>23.94</sub> | <sub>13.26</sub> | <sub>5.2</sub> | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=2`</sub>
| <sub>CREPS</sub> | <sub>8</sub> | <sub>13.04</sub> | <sub>7.32</sub>  | <sub>3.1</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=2`</sub>

#### 512x512 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :---------------: | :---------------: | :------------: | :--
| <sub>CREPS</sub> | <sub>1</sub> | <sub>137.33</sub> | <sub>90.25</sub>  | <sub>7.8</sub> | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=8 --batch-gpu=8 --snap=10`</sub>
| <sub>CREPS</sub> | <sub>2</sub> | <sub>69.65</sub>  | <sub>45.42</sub>  | <sub>7.7</sub> | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=8 --batch-gpu=8 --snap=20`</sub>
| <sub>CREPS</sub> | <sub>4</sub> | <sub>34.88</sub>  | <sub>22.81</sub>  | <sub>7.6</sub> | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=8`</sub>
| <sub>CREPS</sub> | <sub>8</sub> | <sub>18.47</sub>  | <sub>12.29</sub>  | <sub>4.3</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=8`</sub>

#### 1024x1024 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :---------------: | :---------------: | :-------------: | :--
| <sub>CREPS</sub> | <sub>1</sub> | <sub>221.85</sub> | <sub>156.91</sub> | <sub>7.0</sub>  | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5`</sub>
| <sub>CREPS</sub> | <sub>2</sub> | <sub>113.44</sub> | <sub>79.16</sub>  | <sub>6.8</sub>  | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=32 --batch-gpu=4 --snap=10`</sub>
| <sub>CREPS</sub> | <sub>4</sub> | <sub>57.04</sub>  | <sub>39.62</sub>  | <sub>6.7</sub>  | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=32 --batch-gpu=4 --snap=20`</sub>
| <sub>CREPS</sub> | <sub>8</sub> | <sub>28.71</sub>  | <sub>20.01</sub>  | <sub>6.6</sub>  | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=32`</sub>
