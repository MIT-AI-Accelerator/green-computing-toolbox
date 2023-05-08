## Green Computing Toolbox

The Green Computing Toolbox is a python library that enables faster AI/ML model architecture and hyper-parameter searches via [Training Speed Estimation (TSE)](https://doi.org/10.48550/arXiv.2112.03364) and [Loss Curve Gradient Approximation (LCGA)](https://doi.org/10.1109/IPDPSW55747.2022.00123). The toolbox also includes the capability of of NVIDIA GPU power capping via a SLURM plugin and enables the comparison of energy consumed during training of different model architectures.  

The Green Computing Toolbox depends on [Hydra-Zen](https://github.com/mit-ll-responsible-ai/hydra-zen) and the `submitit` plugin and has been tested to work with [SLURM](https://slurm.schedmd.com) managed clusters. Implementation of TSE is from the [LitMatter](https://github.com/ncfrey/litmatter) package. 

## Citation

If you use this tool, please cite the following papers 

```
@INPROCEEDINGS{lcga,
  author={Zhao, Dan and Frey, Nathan C. and Gadepally, Vijay and Samsi, Siddharth},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops}, 
  title={Loss Curve Approximations for Fast Neural Architecture Ranking & Training Elasticity Estimation}, 
  year={2022},
}
```

```
@inproceedings{frey2021scalable,
  title={Scalable Geometric Deep Learning on Molecular Graphs},
  author={Frey, Nathan C and Samsi, Siddharth and McDonald, Joseph and Li, Lin and Coley, Connor W and Gadepally, Vijay},
  booktitle={NeurIPS 2021 AI for Science Workshop},
  year={2021}
}
```


## DISTRIBUTION STATEMENT 

© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
 

The software/firmware is provided to you on an As-Is basis
