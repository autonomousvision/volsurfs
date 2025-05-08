# Volumetric Surfaces (CVPR 2025)

### Representing Fuzzy Geometries with Layered Meshes

### [Project Page](https://autonomousvision.github.io/volsurfs/) | [Paper](https://arxiv.org/pdf/2409.02482) | [Web Demo](https://autonomousvision.github.io/volsurfs/viewer/)


[Stefano Esposito](https://s-esposito.github.io/)<sup>1</sup>,
[Anpei Chen](https://apchenstu.github.io/)<sup>1</sup>, 
[Christian Reiser](https://creiser.github.io/)<sup>1</sup>, 
[Samuel Rota Bulò](https://scholar.google.com/citations?user=484sccEAAAAJ&hl=it)<sup>2</sup>, 
[Lorenzo Porzi](https://scholar.google.it/citations?user=vW1gaVEAAAAJ&hl=it)<sup>2</sup>, 
[Katja Schwarz](https://katjaschwarz.github.io/)<sup>2</sup>, 
[Christian Richardt](https://richardt.name/)<sup>2</sup>, 
[Michael Zollhöfer](https://zollhoefer.com/)<sup>2</sup>, 
[Peter Kontschieder](https://scholar.google.co.uk/citations?user=CxbDDRMAAAAJ&hl=en)<sup>2</sup>,
[Andreas Geiger](https://www.cvlibs.net/)<sup>1</sup>
<br>
<sup>1</sup>[University of Tübingen](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/home/), <sup>2</sup>Meta Reality Labs

<p align="middle">
  <img src="imgs/teaser.png" width="650"/>
</p>



## 🛠️ Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/autonomousvision/volsurfs
cd volsurfs
git submodule update --remote --merge

# Create and activate a conda environment
conda create -n volsurfs python=3.8 cmake=3.31
conda activate volsurfs

# (Optional) Install CUDA toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install PyTorch with CUDA support
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Python dependencies
pip install -r requirements.txt

# Install raytracelib
cd submodules/raytracelib
pip install -e .
cd ../..

# Install NVIDIA APEX
cd submodules/apex
pip install . -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ../..

# Compile and install VolSurfs
pip install ninja
pip install -e . --no-build-isolation
```



## 📁 Datasets

We use the following datasets for training and evaluation:

- [NeRF-Synthetic](scripts/download/blender.sh)
- [DTU](scripts/download/dtu.sh)
- [Shelly](scripts/download/shelly.sh)

Download scripts are located in `scripts/download/` and will place the datasets in the `data/` directory. You can configure dataset paths in `config/paths_config.cfg`.

Example directory structure:
```
data/
├── shelly/
│   ├── khady/
│   ├── kitten/
│   └── ...
├── dtu/
│   ├── dtu_scan24/
│   ├── dtu_scan37/
│   └── ...
├── blender/
│   ├── lego/
│   └── ...
└── ...
```



## 📈 Reproducing Results

To reproduce the main results (5-Mesh) on the **Shelly** dataset, run:

```bash
bash scripts/train_all_shelly.sh
```

Results will be saved in the `runs/` directory. By default, [Weights & Biases](https://wandb.ai/) logging is enabled; you can disable it in `config/train_config.cfg`.



## 📜 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
See the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code as long as you provide proper attribution to the original authors.



## 📚 Citation

If you use this work in your research, please consider citing:

```bibtex
@inproceedings{Esposito2025VolSurfs,
  author    = {Esposito, Stefano and Chen, Anpei and Reiser, Christian and Rota Bulò, Samuel and Porzi, Lorenzo and Schwarz, Katja and Richardt, Christian and Zollhoefer, Michael and Kontschieder, Peter and Geiger, Andreas},
  title     = {Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

```bibtex
@misc{Esposito2025MVD,
  author = {Esposito, Stefano and Geiger, Andreas},
  title  = {MVDatasets: Standardized DataLoaders for 3D Computer Vision},
  year   = {2025},
  url    = {https://github.com/autonomousvision/mvdatasets},
  note   = {GitHub repository}
}
```



## 🙏 Acknowledgements

This repository builds upon [Radu Alexandru Rosu](https://radualexandru.github.io/)'s excellent project [permuto_sdf](https://github.com/RaduAlexandru/permuto_sdf).  
We thank him for sharing his work and providing a strong foundation.
