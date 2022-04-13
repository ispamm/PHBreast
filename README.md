# Multi-View Breast Cancer Classification via Hypercomplex Neural Networks
PHBreast: Official PyTorch repository for Multi-View Breast Cancer Classification via Hypercomplex Neural Networks, _under review_. [[ArXiv preprint]()]

Eleonora Lopez, [Eleonora Grassucci](https://sites.google.com/view/eleonoragrassucci/home-page?authuser=0), Martina Valleriani, and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)


This repository is under construction!


### Pretrained models

| Model                        | Params | Storage Mem | Pretraining mode | Employed Dataset | Weights Link |
|------------------------------|:------:|:-----------:|:----------------:|:----------------:|:------------:|
| PHResNet18 patch classifier  |   5M   |     21MB    | -                | CBIS-DDSM        | [Link](https://drive.google.com/file/d/1FZX_KbOCtBcymZPagrsFEsdVQ_K5zKPx/view?usp=sharing) |
| PHResNet50 patch classifier  |   8M   |     32MB    | -                | CBIS-DDSM        | [Link](https://drive.google.com/file/d/1dZvOvsF1wxj_WhcebHA-z-QLnQGLL4HL/view?usp=sharing) |
| PHResNet18                   |   13M  |     50MB    | Patches          | CBIS-DDSM        | [Link](https://drive.google.com/file/d/1lcyyxSt2ShN5KezhHmCh9B6HpxxtjTBB/view?usp=sharing) |
| PHResNet50                   |   16M  |     62MB    | Patches          | CBIS-DDSM        | [Link](https://drive.google.com/file/d/1P_1h-zyVS_uDterL5AKITvXdcCu_iUKY/view?usp=sharing) |
| PHResNet18                   |   13M  |     50MB    | Patches + CBIS   | INbreast         | [Link](https://drive.google.com/file/d/1J8f5NPcFyQZcubHhR2F_ubmIPoOZT1qu/view?usp=sharing) |
| PHResNet50                   |   16M  |     62MB    | Patches + CBIS   | INbreast         | [Link](https://drive.google.com/file/d/1U3NfKiVejaLP6fN_tdQYwwlUylwG9qpu/view?usp=sharing) |
| PHYSBOnet  (n=2)             |   13M  |     51MB    | Patches          | INbreast         | [Link](https://drive.google.com/file/d/1V0zMzrYDdshHpK7Vxy-qgCD_WzU8IGUx/view?usp=sharing) |
| PHYSBOnet  (n=4, concat)     |   7M   |     27MB    | Patches          | INbreast         | [Link](https://drive.google.com/file/d/1P9GPloZ9MXwlfaa-wa3Bjf-n0j958gpH/view?usp=sharing) |
| PHYSEnet                     |   20M  |     79MB    | Patches          | INbreast         | [Link](https://drive.google.com/file/d/113aMZKeX9vXnhqyzdvIcwbvgf5rlenAJ/view?usp=sharing) |
| PHYSEnet                     |   20M  |     79MB    | Patches + CBIS   | INbreast         | [Link](https://drive.google.com/file/d/1ndXw7h9XdID_JYN9ZrU0U_noY22kOfq8/view?usp=sharing) |


## Cite

Please cite our work if you found it useful:

```
@article{lopez2022phbreast,
      title={Multi-View Breast Cancer Classification via Hypercomplex Neural Networks}, 
      author={Lopez, E. and Grassucci, E. and Valleriani, M. and Comminiello, D.},
      year={2022},
      journal={arXiv preprint:2204.05798}
}
```

## Similar reporitories :busts_in_silhouette:

* [HyperNets](https://github.com/eleGAN23/HyperNets).
* [PHC-GNN](https://github.com/bayer-science-for-a-better-life/phc-gnn).
