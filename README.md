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
> **Abstract:** *A connectional brain template (CBT) is a fingerprint graph-based representation of a population of brain networks, serving as an ’average’ connectome. CBTs are essential for creating representative maps of brain connectivity in both typical and atypical populations, facilitating the identification of deviations from healthy brain structures. However, traditional methods for generating CBTs often rely on linear averaging and pairwise relationships, which fail to capture the complex, high-order interactions within brain networks, particularly in multi-view brain networks where the brain is encoded in a set of connectivity matrices (i.e., tensor). To address these limitations, we propose a novel Generative Hypergraph Neural Network (Gen-HNN) for learning hyper connectional brain templates (HCBTs). Gen-HNN leverages hypergraphs to capture higher-order relationships, utilizing hyperedge convolution operations based on the hypergraph Laplacian to process and integrate multi-view brain data into a cohesive HCBT. Our model overcomes the limitations of existing methods by effectively handling non-linear patterns and preserving the topological properties of brain networks. We conducted extensive experiments, demonstrating that Gen-HNN significantly outperforms state-of-the-art methods in terms of both representativeness and discriminative power.*

## Project Structure

The project is organized in a modular way, making it easy to adapt, extend, and modify. Below is an overview of the key files: 

| File                      | Description                                                                                         |
|----------------------------|-----------------------------------------------------------------------------------------------------|
| **`Config.py`**            | Contains configuration settings for the training process, such as dataset paths and model parameters.|
| **`data_helper.py`**       | Provides data preprocessing utilities such as cross-validation, data splitting, and feature extraction. |
| **`Hypergraph_utilities.py`** | Contains functions for hypergraph construction, including Euclidean distance calculations and incidence matrix generation. |
| **`Model.py`**             | Defines the Gen-HNN model architecture using PyTorch, including hypergraph convolution layers and training logic. |

## Installation

Ensure you have Python 3.x installed. To install the required dependencies, use

```bash
pip install -r requirements.txt
```


## Running the model 
To train the Gen-HNN model, run the following command:

```bash
python main.py
```

## Output

After training, the following outputs will be saved in the output/ directory:

    Trained Gen-HNN models for each cross-validation fold.
    Hypergraph-based connectivity matrices (CBTs) generated during the training process.
    Hypergraph incidence matrices for each subject.

  
# Related references

Hypergraph Neural Networks (HGNN): Feng, Y., You, H., Zhang, Z., Ji, R., Gao, Y.: Hypergraph Neural Networks. arXiv e-prints (2018) arXiv:1809.09401 [https://github.com/iMoonLab/HGNN].

Deep Graph Normalizer (DGN): Gurbuz, M.B., Rekik, I.: Deep graph normalizer: a geometric deep learning ap-proach for estimating connectional brain templates. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VII 23, Springer (2020) 155–165 [https://github.com/basiralab/DGN/tree/master]






