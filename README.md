# Byzantine Resilient Decentralized Learning

## Table of Contents
<!-- MarkdownTOC -->
- [General Information](#introduction)
- [BRIDGE Experiments](#bridge)
- [ByRDiE Experiments](#byrdie)
- [Contributors](#contributors)
<!-- /MarkdownTOC -->

<a name="introduction"></a>
# General Information
This repo includes implementations of **Byzantine-resilient Decentralized Gradient Descent(BRIDGE)** and **Byzantine-resilient Distributed Coordinate Descent for decentralized learning(ByRDiE)** to simulate decentralized learning of a linear multiclassifier on the MNIST dataset using a 1 layer neural network in the presence of Byzantine nodes, similar to the decentralized experiment found in [Adversary-resilient Distributed and Decentralized
Statistical Inference and Machine Learning](https://ieeexplore.ieee.org/document/9084329)

## License and Citation
The code in this repo is being released under the GNU General Public License v3.0; please refer to the [LICENSE](./LICENSE) file in the repo for detailed legalese pertaining to the license. In particular, if you use any part of this code then you must cite both the original paper as well as this codebase as follows:

**Paper Citations:** 
- Z. Yang, A. Gang and W. U. Bajwa, "Adversary-Resilient Distributed and Decentralized Statistical Inference and Machine Learning: An Overview of Recent Advances Under the Byzantine Threat Model," in IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 146-159, May 2020, doi: 10.1109/MSP.2020.2973345.
- Yang, Zhixiong, and Waheed U. Bajwa. "BRIDGE: Byzantine-resilient decentralized gradient descent." arXiv preprint arXiv:1908.08098 (2019).
- Z. Yang and W. U. Bajwa, "ByRDiE: Byzantine-Resilient Distributed Coordinate Descent for Decentralized Learning," in IEEE Transactions on Signal and Information Processing over Networks, vol. 5, no. 4, pp. 611-627, Dec. 2019, doi: 10.1109/TSIPN.2019.2928176.

**Codebase Citation:** J. Shenouda, Z. Yang, W. U. Bajwa, "Codebase---Adversary-Resilient Distributed and Decentralized Statistical Inference and Machine Learning: An Overview of Recent Advances Under the Byzantine Threat Model," GitHub Repository, 2020

For all these experiments we setup a decentralized architecture with 20 nodes and 2,000 training samples from the MNIST dataset allocated to each node. 

## Computing Environment
All of our computational experiments were carried out on a Linux high-performance computing (HPC) cluster provided by the Rutgers Office of Advanced Research Computing; specifically, all of our experiments were run on:

Lenovo NextScale nx360 servers

- 2 x 12-core Intel Xeon E5-2680 v3 "Haswell" processors
- 128 GB RAM
- 1 TB local scratch disk

However we only allocated 4GB of RAM when submitting our jobs. 

## Requirements and Dependencies
To reproduce the environment with necessary dependencies create a conda environment using the `environment.yml` provided.

```
conda env create -f environment.yml 
```

## Data
All MNIST data we used can be found in the `./data` folder the `./data/MNIST/raw` contains the raw MNIST data and we used the `./data/MNIST_read.py` script to read the data into numpy arrays which are then pickled for use in the algorithms.

<a name="bridge"></a>
# BRIDGE
We performed decentralized learning using BRIDGE and some of its variants based on distributed learning screening methods, namely Median, Krum and Bulyan. To train the one layer neural network on MNIST with BRIDGE or its variants run the `dec_BRIDGE.py` script.
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

Example: BRIDGE defending against 2 Byzantine nodes with no faulty nodes.

`python dec_BRIDGE.py 0 -b=2 -s=BRIDGE`

With `2` faulty nodes

`python dec_BRIDGE.py 0 -b=2 -gb=True -s=BRIDGE`

Run 10 of those in parallel with `monte_trial` varying between 0-9 to run 10 independent trials of the experiment. 

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
Example: ByRDiE defending against 2 Byzantine nodes with no faulty nodes
`python dec_ByRDiE.py 0 -b=2`

with `2` faulty nodes

`python dec_ByRDiE.py 0 -b=2 -gb=True`

Run 10 of those in parallel with `monte_trial` varying between 0-9 to run 10 independent trials of the experiment. 

# Plotting

All results get saved in `./result` folder, you run 10 independent trials you can then run the `plot.py` script to plot accuracy vs. number of scalar broadcasts.

The `plot.py` script can be easily modified to run only 1 trial of any of the algorithms instead.

**Note:** Running this code will not perfectly reproduce the results obtained in the paper mentioned above, rather this provides an implementation based on the original codebase.

# Contributors
The original implementation was provided by the author of the paper:

[Zhixiong Yang](https://www.linkedin.com/in/zhixiong-yang-67139152/)

The publicization and refactoring of the code was made possible by:

[Joseph Shenouda](https://github.com/joeshenouda)
