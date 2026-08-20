# Method and data references

This repository contains independent adaptations of established traffic
classification, backdoor attack, and backdoor detection methods. The
implementations are adapted to the packet-sequence representation.

## Data provenance

The preprocessed demo splits in `data/` are derived from the CSTNET-TLS 1.3
dataset introduced with ET-BERT.

- Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi, and Jing Yu.
  “ET-BERT: A Contextualized Datagram Representation with Pre-training
  Transformers for Encrypted Traffic Classification.” *Proceedings of the ACM
  Web Conference 2022 (WWW '22)*, pp. 633–642, 2022.
  https://doi.org/10.1145/3485447.3512217

## Traffic classifier

- Chang Liu, Longtao He, Gang Xiong, Zigang Cao, and Zhen Li. “FS-Net: A Flow
  Sequence Network for Encrypted Traffic Classification.” *IEEE INFOCOM 2019 –
  IEEE Conference on Computer Communications*, pp. 1171–1179, 2019.
  https://doi.org/10.1109/INFOCOM.2019.8737507

## Attack baselines

The UAP baseline is related both to the original formulation of universal
adversarial perturbations and to their packet-padding application in encrypted
traffic classification.

- Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. “BadNets: Identifying
  Vulnerabilities in the Machine Learning Model Supply Chain.” arXiv:1708.06733,
  2017. https://arxiv.org/abs/1708.06733

- Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal
  Frossard. “Universal Adversarial Perturbations.” *Proceedings of the IEEE
  Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 86–94,
  2017. https://doi.org/10.1109/CVPR.2017.17

- John T. Holodnak, Olivia M. Brown, Jason T. Matterer, and Andrew Lemke.
  “Backdoor Poisoning of Encrypted Traffic Classifiers.” *2022 IEEE
  International Conference on Data Mining Workshops (ICDMW)*, pp. 577–585,
  2022. https://doi.org/10.1109/ICDMW58026.2022.00080

- Rui Ning, Chunsheng Xin, and Hongyi Wu. “TrojanFlow: A Neural Backdoor Attack
  to Deep Learning-based Network Traffic Classifiers.” *IEEE INFOCOM 2022 –
  IEEE Conference on Computer Communications*, pp. 1429–1438, 2022.
  https://doi.org/10.1109/INFOCOM48880.2022.9796878

## Backdoor defenses

- Di Tang, XiaoFeng Wang, Haixu Tang, and Kehuan Zhang. “Demon in the Variant:
  Statistical Analysis of DNNs for Robust Backdoor Contamination Detection.”
  *30th USENIX Security Symposium (USENIX Security 21)*, pp. 1541–1558, 2021.
  https://www.usenix.org/conference/usenixsecurity21/presentation/tang-di

- Wanlun Ma, Derui Wang, Ruoxi Sun, Minhui Xue, Sheng Wen, and Yang Xiang. “The
  ‘Beatrix’ Resurrections: Robust Backdoor Detection via Gram Matrices.” *30th
  Annual Network and Distributed System Security Symposium (NDSS 2023)*, 2023.
  https://doi.org/10.14722/ndss.2023.23069

- Xiaoxing Mo, Yechao Zhang, Leo Yu Zhang, Wei Luo, Nan Sun, Shengshan Hu,
  Shang Gao, and Yang Xiang. “Robust Backdoor Detection for Deep Learning via
  Topological Evolution Dynamics.” *2024 IEEE Symposium on Security and Privacy
  (SP)*, pp. 2048–2066, 2024.
  https://doi.org/10.1109/SP54263.2024.00174
