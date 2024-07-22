##### Table of contents
1. [Requirements](#requirements)
2. [Getting Started](#getting-started)
3. [Using networks from Python](#using-networks-from-python)
4. [Preparing datasets](#preparing-datasets)
5. [Training](#training)
6. [Quality Metrics](#quality-metrics)
7. [Contacts](#Contacts)

# Efficient Scale-Invariant Generator with Column-Row Entangled Pixel Synthesis
<a href="https://thuanz123.github.io/creps"><img src="https://img.shields.io/badge/WEBSITE-Visit%20project%20page-blue?style=for-the-badge"></a>
<a href="https://arxiv.org/abs/2303.14157"><img src="https://img.shields.io/badge/arxiv-2303.14157-red?style=for-the-badge"></a>

[Thuan Hoang Nguyen](https://thuanz123.github.io/),
[Thanh Van Le](https://github.com/Luvata),
[Anh Tran](https://scholar.google.com/citations?user=FYZ5ODQAAAAJ)<br>
VinAI Research, Vietnam

> **Abstract:** 
Any-scale image synthesis offers an efficient and scalable solution to synthesize photo-realistic images at any scale, even going beyond 2K resolution. However, existing GAN-based solutions depend excessively on convolutions and a hierarchical architecture, which introduce inconsistency and the "texture sticking" issue when scaling the output resolution. From another perspective, INR-based generators are scale-equivariant by design, but their huge memory footprint and slow inference hinder these networks from being adopted in large-scale or real-time systems. In this work, we propose **C**olumn-**R**ow **E**ntangled **P**ixel **S**ynthesis (**CREPS**), a new generative model that is both efficient and scale-equivariant without using any spatial convolutions or coarse-to-fine design. To save memory footprint and make the system scalable, we employ a novel bi-line representation that decomposes layer-wise feature maps into separate "thick" column and row encodings. Experiments on various datasets, including FFHQ, LSUN-Church, MetFaces, and Flickr-Scenery, confirm CREPS' ability to synthesize scale-consistent and alias-free images at any arbitrary resolution with proper training and inference speed.

![teaser.png](./assets/teaser.png)

Details of the model architecture and experimental results can be found in [our following paper](https://arxiv.org/abs/2303.14157).
```bibtex
@inproceedings{thuan2023creps,
  title={Efficient Scale-Invariant Generator with Column-Row Entangled Pixel Synthesis},
  author={Thuan Hoang Nguyen, Thanh Van Le, Anh Tran},
  year={2023},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
**Please CITE** our paper whenever our model implementation is used to help produce published results or incorporated into other software.

## Additional material

- [CREPS pre-trained models](https://drive.google.com/drive/folders/19ntlAXD7MwYqN7OPVuBcZgaC_yFCwpk7?usp=share_link)
- [StyleGAN3 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3) compatible with this codebase
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan3-t-ffhq-1024x1024.pkl`, `stylegan3-t-ffhqu-1024x1024.pkl`, `stylegan3-t-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-r-ffhq-1024x1024.pkl`, `stylegan3-r-ffhqu-1024x1024.pkl`, `stylegan3-r-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-t-metfaces-1024x1024.pkl`, `stylegan3-t-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-r-metfaces-1024x1024.pkl`, `stylegan3-r-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-t-afhqv2-512x512.pkl`</sub><br>
  > <sub>`stylegan3-r-afhqv2-512x512.pkl`</sub><br>
- [StyleGAN2 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan2) compatible with this codebase
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan2-ffhq-1024x1024.pkl`, `stylegan2-ffhq-512x512.pkl`, `stylegan2-ffhq-256x256.pkl`</sub><br>
  > <sub>`stylegan2-ffhqu-1024x1024.pkl`, `stylegan2-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan2-metfaces-1024x1024.pkl`, `stylegan2-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan2-afhqv2-512x512.pkl`</sub><br>
  > <sub>`stylegan2-afhqcat-512x512.pkl`, `stylegan2-afhqdog-512x512.pkl`, `stylegan2-afhqwild-512x512.pkl`</sub><br>
  > <sub>`stylegan2-brecahad-512x512.pkl`, `stylegan2-cifar10-32x32.pkl`</sub><br>
  > <sub>`stylegan2-celebahq-256x256.pkl`, `stylegan2-lsundog-256x256.pkl`</sub><br>

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using Tesla V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required?  See [Troubleshooting](./assets/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your CREPS Python environment:
  - `conda env create -f environment.yml`
  - `conda activate creps`
* Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

See [Troubleshooting](./assets/troubleshooting.md) for help on common installation and run-time problems.

## Getting started

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames or URLs:

```.bash
# Generate an image using pre-trained FFHQ model.
python gen_images.py --outdir=out --trunc=1 --seeds=2 \
    --network=creps-ffhq-512x512.pkl --scale=2

# Render a 4x2 grid of interpolations for seeds 0 through 31.
python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \
    --network=creps-ffhq-512x512.pkl
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.

**Docker**: You can run the above curated image example using Docker as follows:

```.bash
# Build the creps:latest image
docker build --tag creps .

# Run the gen_images.py script using Docker:
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    creps \
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \
         --network=creps-ffhq-512x512.pkl
```

Note: The Docker image requires NVIDIA driver release `r470` or later.

The `docker run` invocation may look daunting, so let's unpack its contents here:

- `--gpus all -it --rm --user $(id -u):$(id -g)`: with all GPUs enabled, run an interactive session with current user's UID/GID to avoid Docker writing files as root.
- ``-v `pwd`:/scratch --workdir /scratch``: mount current running dir (e.g., the top of this git repo on your host machine) to `/scratch` in the container and use that as the current working dir.
- `-e HOME=/scratch`: let PyTorch and CREPS code know where to cache temporary files such as pre-trained models and custom PyTorch extension build results. Note: if you want more fine-grained control, you can instead set `TORCH_EXTENSIONS_DIR` (for custom extensions build dir) and `DNNLIB_CACHE_DIR` (for pre-trained model download cache). You want these cache dirs to reside on persistent volumes so that their contents are retained across multiple `docker run` invocations.

## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.

The generator consists of two submodules, `G.mapping` and `G.synthesis`, that can be executed separately. They also support various additional options:

```python
w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)
```

Please refer to [`gen_images.py`](./gen_images.py) for complete code example.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./assets/dataset-tool-help.txt) for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

**FFHQ**: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and create a zip archive using `dataset_tool.py`:

```.bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-1024x1024.zip

# Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-256x256.zip \
    --resolution=256x256
```

See the [FFHQ README](https://github.com/NVlabs/ffhq-dataset) for information on how to obtain the unaligned FFHQ dataset images. Use the same steps as above to create a ZIP archive for training and validation.

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and create a ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/metfaces/images --dest=~/datasets/metfaces-1024x1024.zip
```

See the [MetFaces README](https://github.com/NVlabs/metfaces-dataset) for information on how to obtain the unaligned MetFaces dataset images. Use the same steps as above to create a ZIP archive for training and validation.

**AFHQv2**: Download the [AFHQv2 dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and create a ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/afhqv2 --dest=~/datasets/afhqv2-512x512.zip
```

Note that the above command creates a single combined dataset using all images of all three classes (cats, dogs, and wild animals), matching the setup used in the CREPS paper. Alternatively, you can also create a separate dataset for each class:

```.bash
python dataset_tool.py --source=~/downloads/afhqv2/train/cat --dest=~/datasets/afhqv2cat-512x512.zip
python dataset_tool.py --source=~/downloads/afhqv2/train/dog --dest=~/datasets/afhqv2dog-512x512.zip
python dataset_tool.py --source=~/downloads/afhqv2/train/wild --dest=~/datasets/afhqv2wild-512x512.zip
```

## Training

You can train new networks using `train.py`. For example:

```.bash
# Train CREPS for FFHQ using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=creps --data=~/datasets/ffhq-512x512.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1

# Fine-tune CREPS for MetFaces using 1 GPU, starting from the pre-trained FFHQ pickle.
python train.py --outdir=~/training-runs --cfg=creps --data=~/datasets/metfaces-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=creps-ffhq-512x512.pkl
```

Besides CREPS, our code also supports StyleGAN3 and StyleGAN2. Note that the result quality and training time depend heavily on the exact set of options. The most important ones (`--gpus`, `--batch`, and `--gamma`) must be specified explicitly, and they should be selected with care. See [`python train.py --help`](./assets/train-help.txt) for the full list of options and [Training configurations](./assets/configs.md) for general guidelines &amp; recommendations, along with the expected training speed &amp; memory usage in different scenarios.

The results of each training run are saved to a newly created directory, for example `~/training-runs/00000-creps-afhqv2-512x512-gpus8-batch32-gamma8.2`. The training loop exports network pickles (`network-snapshot-<KIMG>.pkl`) and random image grids (`fakes<KIMG>.png`) at regular intervals (controlled by `--snap`). For each exported pickle, it evaluates FID (controlled by `--metrics`) and logs the result in `metric-fid50k_full.jsonl`. It also records various statistics in `training_stats.jsonl`, as well as `*.tfevents` if TensorBoard is installed.

## Quality metrics

By default, `train.py` automatically computes FID for each network pickle exported during training. We recommend inspecting `metric-fid50k_full.jsonl` (or TensorBoard) at regular intervals to monitor the training progress. When desired, the automatic computation can be disabled with `--metrics=none` to speed up the training slightly.

Additional quality metrics can also be computed after the training:

```.bash
# Previous training run: look up options automatically, save result to JSONL file.
python calc_metrics.py --metrics=fid50k_full \
    --network=~/training-runs/creps-mydataset/network-snapshot-000000.pkl

# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \
    --network=creps-ffhq-512x512.pkl --scale=2
```

The first example looks up the training configuration and performs the same operation as if `--metrics=fid50k_full` had been specified during training. The second example downloads a pre-trained network pickle, in which case the values of `--data` and `--mirror` must be specified explicitly.

Note that the metrics can be quite expensive to compute (up to 1h), and many of them have an additional one-off cost for each new dataset (up to 30min). Also note that the evaluation is done using a different random seed each time, so the results will vary if the same metric is computed multiple times.

Recommended metrics:
* `fid50k_full`: Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset.
* `kid50k_full`: Kernel inception distance<sup>[2]</sup> against the full dataset.
* `pr50k3_full`: Precision and recall<sup>[3]</sup> againt the full dataset.
* `ppl2_wend`: Perceptual path length<sup>[4]</sup> in W, endpoints, full image.
* `eqt50k_int`: Equivariance<sup>[5]</sup> w.r.t. integer translation (EQ-T).
* `eqt50k_frac`: Equivariance w.r.t. fractional translation (EQ-T<sub>frac</sub>).
* `eqr50k`: Equivariance w.r.t. rotation (EQ-R).

Legacy metrics:
* `fid50k`: Fr&eacute;chet inception distance against 50k real images.
* `kid50k`: Kernel inception distance against 50k real images.
* `pr50k3`: Precision and recall against 50k real images.
* `is50k`: Inception score<sup>[6]</sup> for CIFAR-10.

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018
5. [Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3), Karras et al. 2021
6. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016

## Acknowledgments
This repo is built upon [StyleGAN3](https://github.com/NVlabs/stylegan3) and we thank the authors for their great works and efforts to release source code. Furthermore, a special "thank you" to [CIPS](https://github.com/advimman/CIPS) and [AnyresGAN](https://github.com/chail/anyres-gan), which heavily inspire our work.

## Contacts
If you have any questions, please drop an email to _v.thuannh5@vinai.io_ or open an issue in this repository.
