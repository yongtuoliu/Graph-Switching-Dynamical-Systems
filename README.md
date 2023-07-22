# Graph-Switching-Dynamical-Systems

This is the official code repository for the papers **"Graph Switching Dynamical Systems"** (ICML 2023), by Yongtuo Liu, Sara Magliacane, Miltiadis Kofinas, and Efstratios Gavves.

## Requirements

The code was originally written for PyTorch v1.7.1 and Python 3.7.11 Higher versions of PyTorch and Python are expected to work as well. Other packages can be installed automatically by dependence.

## Datasets

ODE-driven Particle Dataset and Salsa-couple Dancing Dataset are used in our experiments. Both of them are created by ourselves. As ODE-driven Particle Dataset is a synthetic dataset, we create it by codes. The generation codes are in Folder Data_generation. Salsa-couple Dancing Dataset is in the following link.

## Models

Experimental models are in Folder Models.

## Citation

If you use this code or find it otherwise helpful, please consider citing our work:
```bibtex
@InProceedings{pmlr-v202-liu23z,
  title = 	 {Graph Switching Dynamical Systems},
  author =       {Liu, Yongtuo and Magliacane, Sara and Kofinas, Miltiadis and Gavves, Efstratios},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {21867--21883},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/liu23z/liu23z.pdf},
  url = 	 {https://proceedings.mlr.press/v202/liu23z.html},
  abstract = 	 {Dynamical systems with complex behaviours, e.g. immune system cells interacting with a pathogen, are commonly modelled by splitting the behaviour in different regimes, or <em>modes</em>, each with simpler dynamics, and then learn the switching behaviour from one mode to another. To achieve this, Switching Dynamical Systems (SDS) are a powerful tool that automatically discovers these modes and mode-switching behaviour from time series data. While effective, these methods focus on <em>independent objects</em>, where the modes of one object are independent of the modes of the other objects. In this paper, we focus on the more general <em>interacting object</em> setting for switching dynamical systems, where the per-object dynamics also depend on an unknown and dynamically changing subset of other objects and their modes. To this end, we propose a novel graph-based approach for switching dynamical systems, GRAph Switching dynamical Systems (GRASS), in which we use a dynamic graph to characterize interactions between objects and learn both intra-object and inter-object mode-switching behaviour. For benchmarking, we create two new datasets, a synthesized ODE-driven particles dataset and a real-world Salsa-couple dancing dataset. Experiments show that GRASS can consistently outperforms previous state-of-the-art methods. We will release code and data after acceptance.}
}
```


### Contact

If you have questions or found a bug, feel free to open a github issue or send a mail to y.liu6@uva.nl. 
