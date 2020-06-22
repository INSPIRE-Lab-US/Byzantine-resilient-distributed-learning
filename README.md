# Byzantine Resilient Decentralized Learning

## Table of Contents
<!-- MarkdownTOC -->
- [General Information](#introduction)
- [BRIDGE Experiments](#bridge)
- [ByRDiE Experiments](#byrdie)
- [Plotting](#plotting)
- [Contributors](#contributors)
<!-- /MarkdownTOC -->

<a name="introduction"></a>
# General Information
This repo includes implementations of **Byzantine-resilient Decentralized Gradient Descent(BRIDGE)** and its variants as well as **Byzantine-resilient Distributed Coordinate Descent for decentralized learning(ByRDiE)** to perform decentralized learning in the presence of Byzantine nodes. Specifically, this codebase impelements the decentralized learning experiment found in [Adversary-resilient Distributed and Decentralized
Statistical Inference and Machine Learning](https://ieeexplore.ieee.org/document/9084329)

## License and Citation
The code in this repo is being released under the GNU General Public License v3.0; please refer to the [LICENSE](./LICENSE) file in the repo for detailed legalese pertaining to the license. In particular, if you use any part of this code then you must cite both the original papers as well as this codebase as follows:

**Paper Citations:** 
- Z. Yang, A. Gang and W. U. Bajwa, "Adversary-Resilient Distributed and Decentralized Statistical Inference and Machine Learning: An Overview of Recent Advances Under the Byzantine Threat Model," in IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 146-159, May 2020, doi: [10.1109/MSP.2020.2973345](https://doi.org/10.1109/MSP.2020.2973345).
- Yang, Zhixiong, and Waheed U. Bajwa. "BRIDGE: Byzantine-resilient decentralized gradient descent." arXiv preprint arXiv:1908.08098 (2019).
- Z. Yang and W. U. Bajwa, "ByRDiE: Byzantine-Resilient Distributed Coordinate Descent for Decentralized Learning," in IEEE Transactions on Signal and Information Processing over Networks, vol. 5, no. 4, pp. 611-627, Dec. 2019, doi: [10.1109/TSIPN.2019.2928176](https://doi.org/10.1109/TSIPN.2019.2928176).

**Codebase Citation:** J. Shenouda, Z. Yang, W. U. Bajwa, "Codebase---Adversary-Resilient Distributed and Decentralized Statistical Inference and Machine Learning: An Overview of Recent Advances Under the Byzantine Threat Model," GitHub Repository, 2020

## Summary of Experiments
In these experiments we simulate a decentralized network to learn a linear multiclassifier on the MNIST dataset using a one layer neural network implemented in TensorFlow. Our network consists of twenty nodes each assigned two thousand training samples from the MNIST dataset. Similar to the paper we conduct two experiments.

1. Train the neural network using Distributed Gradient Descent (DGD), ByRDiE, BRIDGE and the variants of BRIDGE namely, Median, Krum and Bulyan defending against two byzantine nodes while no nodes actually undergo byzantine failure. This is the faultless setting and will produce a plot similar to Figure 3a in the paper.

2. Train the neural network using all six methods as above while defending against two byzantine nodes and indeed two nodes undergo byzantine failure communicating random values instead of the actual gradient. This is the faulty setting and will produce a plot similar to Figure 3b in the paper.

For each of these experiments, the faultless and fauly setting, we ran them ten times in parallel and averaged the classification accuracy before plotting.

## Computing Environment
All of our computational experiments were carried out on a Linux high-performance computing (HPC) cluster provided by the Rutgers Office of Advanced Research Computing; specifically, all of our experiments were run on:

Lenovo NextScale nx360 servers

- 2 x 12-core Intel Xeon E5-2680 v3 "Haswell" processors
- 128 GB RAM
- 1 TB local scratch disk

However we only allocated 4GB of RAM when submitting each of our jobs. 

## Requirements and Dependencies
To reproduce the environment with necessary dependencies create a conda environment using the `environment.yml` provided.

```
conda env create -f environment.yml 
```

## Data
The MNIST dataset we used can be found in the `./data` directory. The `./data/MNIST/raw` directory contains the raw MNIST data and the `./data/MNIST_read.py` script reads the data into numpy arrays which are then pickled for use in the experiments. The pickled numpy arrays are already avaliable in the `./data/MNIST/pickled` directory so there is no need to rerun our script in order to perform the experiments.

<a name="bridge"></a>
# BRIDGE Experiments
We performed decentralized learning using BRIDGE and some of its variants based on distributed learning screening methods, namely Median, Krum and Bulyan. To train the one layer neural network on MNIST with BRIDGE or its variants run the `dec_BRIDGE.py` script. When no screening method is training is done with distributed gradient descent and no screening.
```
usage: dec_BRIDGE.py [-h] [-b BYZANTINE] [-gb GOBYZANTINE]
                     [-s {BRIDGE,Median,Krum,Bulyan}]
                     monte_trial

positional arguments:
  monte_trial           Specify which monte carlo trial to run

optional arguments:
  -h, --help            show this help message and exit
  -b BYZANTINE, --byzantine BYZANTINE
                        Number of Byzantine nodes to defend against, if none
                        defaults to 0
  -gb GOBYZANTINE, --goByzantine GOBYZANTINE
                        Boolean to indicate if the specified number of
                        Byzantine nodes actually send out faulty values
  -s {BRIDGE,Median,Krum,Bulyan}, --screening {BRIDGE,Median,Krum,Bulyan}
                        Screening method to use (BRIDGE,Median, Krum, Bulyan),
                        default no screening is done regular gradient descent
```

Example: BRIDGE defending against two Byzantine nodes with no faulty nodes (faultless).

`python dec_BRIDGE.py 0 -b=2 -s=BRIDGE`

With two faulty nodes (faulty)

`python dec_BRIDGE.py 0 -b=2 -gb=True -s=BRIDGE`

Run each of the possible screening methods ten times in parallel by varying `monte_trial` between 0 and 9 for ten independent Monte Carlo trials.

Each Monte Carlo trial ran in about 100 seconds on our machines for all screening methods.

<a name="byrdie"></a>
# ByRDiE Experiments
We performed decentralized learning using ByRDiE, both in the faultless setting and in the presence of Byzantine nodes. To train the one layer neural network on MNIST with ByRDiE run the `dec_ByRDiE.py` script.
```
usage: dec_ByRDiE.py [-h] [-b BYZANTINE] [-gb GOBYZANTINE] monte_trial

positional arguments:
  monte_trial           Specify which monte carlo trial to run

optional arguments:
  -h, --help            show this help message and exit
  -b BYZANTINE, --byzantine BYZANTINE
                        Number of Byzantine nodes to defend against, if none
                        defaults to 0
  -gb GOBYZANTINE, --goByzantine GOBYZANTINE
                        Boolean to indicate if the specified number of
                        Byzantine nodes actually send out faulty values
```
Example: ByRDiE defending against two byzantine nodes with no faulty nodes
`python dec_ByRDiE.py 0 -b=2`

with two faulty nodes

`python dec_ByRDiE.py 0 -b=2 -gb=True`

Run `dec_ByRDiE.py` ten times in parallel by varying `monte_trial` between 0 and 9 for ten independent Monte Carlo trials.

Each Monte Carlo trial for ByRDiE ran in about two days on our machines.
<a name="plotting"></a>
# Plotting

All results get saved in `./result` folder, after running ten independent trials for each screening method as described above run the `plot.py` script to generate the plots similar to Figure 3 in the paper.

**Note:** Due to a loss in the original implementation of the decentralized Krum and Bulyan screening methods the experiments with these screening methods will not perfectly reproduce the results found in Figure 3 of the paper. Nonetheless the results from the implementations in this codebase are consistent with the discussions and conclusions made in the paper.

# Contributors
The original implementation was provided by the author of the paper:

- [Zhixiong Yang](https://www.linkedin.com/in/zhixiong-yang-67139152/)
- [Arpita Gang](https://www.linkedin.com/in/arpita-gang-41444930/)
- [Waheed U. Bajwa](http://www.inspirelab.us/)

The publicization and reproducibility of the code was made possible by:

- [Joseph Shenouda](https://github.com/joeshenouda)
