# Localized Conformal Prediction for Image Classification with Vision-Language Models
The official repository of the paper *Localized Conformal Prediction for Image
Classification with Vision-Language Models* accepted at EUVIP 2025 (link coming soon).

Authors:
[Tim Bary*](https://scholar.google.com/citations?user=LgS_I5UAAAAJ&hl=en&oi=ao),
[Clément Fuchs*](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=en&oi=ao),
[Benoît Macq](https://scholar.google.com/citations?user=H9pGN70AAAAJ&hl=en&oi=ao).

*Denotes equal contribution.

## Overview
This work explores the use of Localized Conformal Prediction (LCP) in image classification tasks using Vision-Language Models (VLMs) like CLIP. Conformal prediction is a framework that provides prediction sets with guaranteed marginal coverage. LCP additionally tailors prediction sets based on the similarity between test and calibration samples. We provide an open source implementation of the algorithm presented in *Localized conformal prediction: A generalized inference framework for conformal prediction*, L. Guan, for classification as well as an extensive benchmark on 10 datsets using CLIP. 
We show that cosine similarities alone fail to improve significantly over non-local baselines, and that a simple sigmoid transformation of cosine similarity, tuned via cross-validation, leads to smaller, more efficient prediction sets without sacrificing coverage. We benchmark across 10 datasets and 4 VLM backbones, demonstrating consistent gains.

## Datasets
Please follow [DATASETS.md](DATASETS.md) to install the datasets.
You will get a structure with the following dataset names:
```
$DATA/
├── UCF101/
└── Food101/
```

## Reproducing the Results
Use the bash scripts provided in ./scripts. 

## Citation

If you find this repository useful, please consider citing our paper:<!-- Change arXiv number -->
```
@article{bary2025conformal,
  title={Conformal Predictions for Human Action Recognition with Vision-Language Models},
  author={Bary, Tim and Fuchs, Cl{\'e}ment and Macq, Beno{\^i}t}
  journal={arXiv preprint arXiv:2502.06631},
  year={2025}
}
```
as well as 
```
@article{guan2023localized,
  title={Localized conformal prediction: A generalized inference framework for conformal prediction},
  author={Guan, Leying},
  journal={Biometrika},
  volume={110},
  number={1},
  pages={33--50},
  year={2023},
  publisher={Oxford University Press}
}
```
## Contact

For any inquiries, please contact us at [tim.bary@uclouvain.be](mailto:tim.bary@uclouvain.be) and  [clement.fuchs@uclouvain.be](mailto:clement.fuchs@uclouvain.be) or feel free to create an issue.

## Acknowledgments
T.Bary and C.Fuchs are funded by the MedReSyst project, supported by FEDER and the Walloon Region. Part of the computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the Fonds de la Recherche Scientifique de Belgique (F.R.S.-FNRS) under Grant No. 2.5020.11 and by the Walloon Region.
