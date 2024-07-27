# Gen-HNN
We develped a model that generates a hyperconnectional brain template from a set of multiview brain networks 

Please contact mayssa.soussia@gmail.com for inquiries. Thanks.


![Main Figure](Main_Figure.png)

# Introduction
This work  has been accepted in PRIME MICCAI workshop 2024, Marrakech, Morroco. 


> **Generative Hypergraph Neural Network for Multiview Brain Connectivity Fusion **
>
> Mayssa Soussia, Mohamed Ali Mahjoub and Islem Rekik
>
> LATIS Lab, National Engineering School of Sousse, University of Sousse, Tunisia
> 
> BASIRA Lab, Imperial-X and Department of Computing, Imperial College London, London, UK
>
> **Abstract:** *A connectional brain template (CBT) is a fingerprint graph-based representation of a population of brain networks, serving as an ’average’ connectome. CBTs are essential for creating representative maps of brain connectivity in both typical and atypical populations, facilitating the identification of deviations from healthy brain structures. However, traditional methods for generating CBTs often rely on linear averaging and pairwise relationships, which fail to capture the complex, high-order interactions within brain networks, particularly in multi-view brain networks where the brain is encoded in a set of connectivity matrices (i.e., tensor). To address these limitations, we propose a novel Generative Hypergraph Neural Network (Gen-HNN) for learning hyper connectional brain templates (HCBTs). Gen-HNN leverages hypergraphs to capture higher-order relationships, utilizing hyperedge convolution operations based on the hypergraph Laplacian to process and integrate multi-view brain data into a cohesive HCBT. Our model overcomes the limitations of existing methods by effectively handling non-linear patterns and preserving the topological properties of brain networks. We conducted extensive experiments, demonstrating that Gen-HNN significantly outperforms state-of-the-art methods in terms of both representativeness and discriminative power..*

# Related references

Hypergraph Neural Networks (HGNN): Feng, Y., You, H., Zhang, Z., Ji, R., Gao, Y.: Hypergraph Neural Networks. arXiv e-prints (2018) arXiv:1809.09401 [https://github.com/iMoonLab/HGNN].

Deep Graph Normalizer (DGN): Gurbuz, M.B., Rekik, I.: Deep graph normalizer: a geometric deep learning ap-proach for estimating connectional brain templates. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VII 23, Springer (2020) 155–165 [https://github.com/basiralab/DGN/tree/master]






