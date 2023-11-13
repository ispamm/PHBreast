# Multi-View Breast Cancer Classification via Hypercomplex Neural Networks
PHBreast: Official PyTorch repository for Multi-View Breast Cancer Classification via Hypercomplex Neural Networks, _under review_. [[ArXiv preprint](https://arxiv.org/pdf/2204.05798.pdf)]

Eleonora Lopez, [Eleonora Grassucci](https://sites.google.com/uniroma1.it/eleonoragrassucci/home-page), Martina Valleriani, and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)

## Abstract 📑

Traditionally, deep learning-based methods for breast cancer classification perform a single-view analysis. However, radiologists simultaneously analyze all four views that compose a mammography exam, owing to the correlations contained in mammography views, which present crucial information for identifying tumors. In light of this, some studies have started to propose multi-view methods. Nevertheless, in such existing architectures, mammogram views are processed as independent images by separate convolutional branches, thus losing correlations among them. To overcome such limitations, in this paper we propose a novel approach for multi-view breast cancer classification based on parameterized hypercomplex neural networks. Thanks to hypercomplex algebra properties, our networks are able to model, and thus leverage, existing correlations between the different views that comprise a mammogram exam, thus mimicking the reading process performed by clinicians. As a consequence, the proposed method is able to handle the information of a patient altogether without breaking the multi-view nature of the exam. Starting from the proposed hypercomplex approach, we define architectures designed to process two-view exams, namely PHResNets, and four-view exams, i.e., PHYSEnet and PHYSBOnet, with the ability to grasp inter-view correlations in a wide range of clinical use cases.
Through an extensive experimental evaluation conducted with two publicly available datasets, CBIS-DDSM and INbreast, we demonstrate that our parameterized hypercomplex models clearly outperform real-valued counterparts and also state-of-the-art methods, proving that breast cancer classification benefits from the proposed multi-view architecture.

## Usage :information_source:

- `pip install -r requirements.txt`
- Choose the configuration and run the experiment: 

`python main.py --TextArgs=configs/config_name.txt`.

The experiment will be directly tracked on [Weight&Biases](https://wandb.ai/).

## Data :open_file_folder:

You can download the preprocessed data here. Each .zip file contains the training and validation splits. 
To use the provided DataLoader, simply pass  `--train_dir=/path/to/unzipped_folder`. A notebook with preprocessing of INbreast is also available in the utils folder.

| Dataset          | Num views | Storage Mem | Link   |
|------------------|:---------:|:-----------:|:------:|
| CBIS - patches   | 2 views   | 1GB         | [Link](https://drive.google.com/file/d/15jVK-ICQ8c4zKp807q53ds5PEKsvNDzq/view?usp=sharing) |
| CBIS - mass      | 2 views   | 146MB       | [Link](https://drive.google.com/file/d/16H0JbQKecIy8i376--m_ut-PwXduRDNJ/view?usp=sharing) |
| CBIS - mass+calc | 2 views   | 290MB       | [Link](https://drive.google.com/file/d/1pPmFNwFbvDBvzD4Srw-p6Kw__r-gY7U8/view?usp=sharing) |
| INbreast         | 2 views   | 25MB        | [Link](https://drive.google.com/file/d/1dDwH8E-1jg0k5VzpJ8pKiM_2KwRKagZ1/view?usp=sharing) |
| INbreast         | 4 views   | 21MB        | [Link](https://drive.google.com/file/d/1Gn3U6cS1TYQ7N_qDT6awRubawio6_8PV/view?usp=sharing) |

### Training :hammer:

To repeat **two-view** experiments in the paper use the configuration files provided in `configs/`:
- Training patch classifier: `config_phcresnet_cbis_patches.txt`.
- Training whole-image classifier (pretrained on Patches) with CBIS-DDSM: `config_phcresnet_cbis2views.txt`.
- Training whole-image classifier (pretrained on Patches + CBIS) with INbreast pretrained: `config_phcresnet_inbreast2views.txt`.

To repeat **four-view** experiments in the paper use the configuration files provided in `configs/`:
- Training PHYSBOnet (pretrained on Patches) with *n=2*: `config_physbonet_inbreast4views_shared.txt`.
- Training PHYSBOnet (pretrained on Patches) with *n=4*, *concat version*: `config_physbonet_inbreast4views_concat.txt`.
- Training PHYSEnet (pretrained on Patches): `config_physenet_inbreast4views_patches_pretr.txt`.
- Training PHYSEnet (pretrained on Patches + CBIS): `config_physenet_inbreast4views_wholeimg_pretr.txt`.

### Evaluation :electric_plug:

For evaluation simply download the weights and use the same configuration file as above by changing:
- `--model_state=/path/to/weights` with the path of the dowloaded weights.
- `--evaluate_model=True`.

### Visualization :chart_with_upwards_trend:

To visualize the saliency maps and activation maps you can use the notebook Visualize.ipynb.

## Pretrained models :nut_and_bolt:

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

## Similar repositories :busts_in_silhouette:

* [HyperNets](https://github.com/eleGAN23/HyperNets)
* [MHyEEG](https://github.com/ispamm/MHyEEG/)
* [PHC-GNN](https://github.com/bayer-science-for-a-better-life/phc-gnn)
