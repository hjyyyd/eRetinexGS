# eRetinexGS: Retinex Modeling for Low-Light Scene Enhancement via Event Streams and 3D Gaussian Splatting

Official implementation of our CVPR 2026 paper:
> **eRetinexGS: Retinex Modeling for Low-Light Scene Enhancement via Event Streams and 3D Gaussian Splatting**
> 
> Haojie Yan, Zehao Chen, Yan Liu, Shi Gu, Peng Lin, De Ma, Huajin Tang, Qian Zheng†, Gang Pan†
>
> State Key Lab of Brain-Machine Intelligence, Zhejiang University

## Links
- [Project Page](https://zju-bmi-lab.github.io/eRetinexGS-homepage/)
- [Paper(CVPR)](https://openaccess.thecvf.com/content/CVPR2026/papers/Yan_eRetinexGS_Retinex_Modeling_for_Low-Light_Scene_Enhancement_via_Event_Streams_CVPR_2026_paper.pdf)


## Overview

eRetinexGS is an event-guided framework that jointly leverages **low-light frames** and
**event streams** to reconstruct a normal-light radiance field via 3D Gaussian Splatting.
Each Gaussian explicitly models a Retinex decomposition (reflectance `R` and illumination `L`,
with scene radiance `I = R ⊙ L`). An *event-guided reflectance smoothness* prior and a
*confidence-guided complementarity* between events and frames drive a robust decomposition,
enabling both low-light enhancement and novel-view synthesis.

## Abstract

Perception under low illumination remains a major challenge for computer vision systems, as RGB sensors often fail to capture sufficient structural and color information in extremely dark environments. Event cameras, with their high dynamic range and temporal resolution, provide complementary cues that are well suited for such conditions. In this work, we present eRetinexGS, a novel framework that jointly leverages event streams and low-light frames through 3D Gaussian Splatting for scene-level enhancement and reconstruction. Unlike previous approaches that operate on individual frames, eRetinexGS enforces geometric and photometric consistency across multiple views, bridging the gap between degraded images and noisy event signals. By introducing an event-assisted Retinex decomposition and a reflectance–illumination representation within the 3DGS pipeline, our method reconstructs normal-light radiance fields with fine-grained details and accurate color. Extensive experiments on both synthetic and real datasets demonstrate that eRetinexGS achieves state-of-the-art performance in low-light scene enhancement while maintaining real-time rendering capability.



## Citation

```bibtex
@inproceedings{yan2026eretinexgs,
  title     = {eRetinexGS: Retinex Modeling for Low-Light Scene Enhancement via Event Streams and 3D Gaussian Splatting},
  author    = {Yan, Haojie and Chen, Zehao and Liu, Yan and Gu, Shi and Lin, Peng and Ma, De and Tang, Huajin and Zheng, Qian and Pan, Gang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

## Acknowledgements

This code builds on [GS-IR](https://github.com/lzhnb/GS-IR),
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting),
and [nvdiffrast](https://github.com/NVlabs/nvdiffrast). We thank the authors for their work.
