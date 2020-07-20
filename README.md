# Byzantine-resilient Decentralized Machine Learning: Codebase for ByRDiE, BRIDGE, and Variants

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
This repo provides implementations of **Byzantine-resilient Distributed Coordinate Descent for Decentralized Learning (ByRDiE)**, **Byzantine-resilient Decentralized Gradient Descent (BRIDGE)**, and different variants of the BRIDGE algorithm. In addition, it includes code to implement decentralized machine learning in the presence of Byzantine (malicious) nodes. The codebase in particular can be used to reproduce the decentralized learning experiments reported in the overview paper entitled "[Adversary-resilient Distributed and Decentralized Statistical Inference and Machine Learning](https://ieeexplore.ieee.org/document/9084329)" that appeared in IEEE Signal Processing Magazine in May 2020.

## License and Citation
The code in this repo is being released under the GNU General Public License v3.0; please refer to the [LICENSE](./LICENSE) file in the repo for detailed legalese pertaining to the license. In particular, if you use any part of this code then you must cite both the original papers as well as this codebase as follows:

**Paper Citations:** 

- Z. Yang and W.U. Bajwa, "ByRDiE: Byzantine-resilient distributed coordinate descent for decentralized learning," IEEE Trans. Signal Inform. Proc. over Netw., vol. 5, no. 4, pp. 611-627, Dec. 2019; doi: [10.1109/TSIPN.2019.2928176](https://doi.org/10.1109/TSIPN.2019.2928176).
- Z. Yang and W.U. Bajwa, "BRIDGE: Byzantine-resilient decentralized gradient descent," arXiv preprint, Aug. 2019; [arXiv:1908.08098](https://arxiv.org/abs/1908.08098).
- Z. Yang, A. Gang, and W.U. Bajwa, "Adversary-resilient distributed and decentralized statistical inference and machine learning: An overview of recent advances under the Byzantine threat model," IEEE Signal Processing Mag., vol. 37, no. 3, pp. 146-159, May 2020; doi: [10.1109/MSP.2020.2973345](https://doi.org/10.1109/MSP.2020.2973345).

**Codebase Citation:** J. Shenouda, Z. Yang, and W.U. Bajwa, "Codebase---Adversary-resilient distributed and decentralized statistical inference and machine learning: An overview of recent advances under the Byzantine threat model," GitHub Repository, 2020; doi: [TBD](#).

## Summary of Experiments
The codebase uses implementations of ByRDiE, BRIDGE, and BRIDGE variants to generate results for Byzantine-resilient decentralized learning. The generated results correspond to experiments in which we simulate a decentralized network that trains a linear multiclass classifier on the [MNSIT dataset](http://yann.lecun.com/exdb/mnist/) using a one-layer neural network that is implemented in TensorFlow. The network consists of twenty nodes, with each node assigned two thousand training samples from the MNIST dataset. Similar to the overview paper (Yang et al., 2020), the codebase provides two sets of experiments:

1. Train the neural network using Distributed Gradient Descent (DGD), ByRDiE, BRIDGE, and three variants of BRIDGE, namely, BRIDGE--Median, BRIDGE--Krum and BRIDGE--Bulyan, with the Byzantine-resilient algorithms defending against at most two Byzantine nodes while no nodes actually undergo Byzantine failure. This is the faultless setting and the code produces a plot similar to Figure 3(a) in the paper (Yang et al., 2020) in this case.
2. Train the neural network using the six methods as above, with the Byzantine-resilient algorithms defending against at most two Byzantine nodes and exactly two nodes undergo Byzantine failure and communicate random values instead of the actual gradient to their neighbors. This is the faulty setting and the code produces a plot similar to Figure 3(b) in the paper (Yang et al., 2020) in this case.

For experiments in both the faultless and the faulty setting, we ran ten Monte Carlo trials in parallel and averaged the classification accuracy before plotting.

## Summary of Code
The `dec_BRIDGE.py` and `dec_ByRDiE.py` serve as the "driver" or "main" files where we set up the experiments and call the necessary functions to learn the machine learning model in a decentralized manner. The actual implementations of the various screenings methods (ByRDiE, BRIDGE, and variants of BRIDGE) are carried out in the `DecLearning.py` module. While these specific implementations are written for the particular case of training with a single-layer neural network using TensorFlow, the core of these implementations can be easily adapted for other machine learning problems.

## Computing Environment
All of our computational experiments were carried out on a Linux high-performance computing (HPC) cluster provided by the Rutgers Office of Advanced Research Computing; specifically, all of our experiments were run on:

Lenovo NextScale nx360 servers:

- 2 x 12-core Intel Xeon E5-2680 v3 "Haswell" processors
- 128 GB RAM
- 1 TB local scratch disk

However, we only allocated 4GB of RAM when submitting each of our jobs. 

## Requirements and Dependencies
This code is written in Python and uses TensforFlow.  To reproduce the environment with necessary dependencies needed for running of the code in this repo, we recommend that the users create a `conda` environment using the `environment.yml` YAML file that is provided in the repo. Assuming the conda management system is installed on the user's system, this can be done using the following:

```
$ conda env create -f environment.yml
```

In the case users don't have conda installed on their system, they should check out the `environment.yml` file for the appropriate version of Python as well as the necessary dependencies with their respective versions needed to run the code in the repo.

## Data
The MNIST dataset we used in our experiments can be found in the `./data` directory. The `./data/MNIST/raw` directory contains the raw MNIST data, as available from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), while the `./data/MNIST_read.py` script reads the data into `numpy` arrays that are then *pickled* for use in the experiments. The pickled numpy arrays are already available in the `./data/MNIST/pickled` directory, so there is no need to rerun our script in order to perform the experiments.

<a name="bridge"></a>
# BRIDGE Experiments
We performed decentralized learning using BRIDGE and some of its variants based on distributed learning screening methods, namely Median, Krum and Bulyan. To train the one-layer neural network on MNIST with BRIDGE or its variants, run the `dec_BRIDGE.py` script. When no screening method is selected, training is done with distributed gradient descent (DGD) without screening. Each Monte Carlo trial ran in about one hundred seconds on our machines for each of the screening methods.

```
usage: dec_BRIDGE.py [-h] [-b BYZANTINE] [-gb GOBYZANTINE]
                     [-s {BRIDGE,Median,Krum,Bulyan}]
                     monte_trial

positional arguments:
  monte_trial           A number between 0 and 9 to indicate which
  					  Monte Carlo trial to run

optional arguments:
  -h, --help            Show this help message and exit
  -b BYZANTINE, --byzantine BYZANTINE
                        Maximum number of Byzantine nodes to defend
                        against; if none then it defaults to 0
  -gb GOBYZANTINE, --goByzantine GOBYZANTINE
                        Boolean to indicate if the specified number of
                        Byzantine nodes actually send out faulty values
  -s {BRIDGE,Median,Krum,Bulyan}, --screening {BRIDGE,Median,Krum,Bulyan}
                        Screening method to use (BRIDGE,Median,Krum,Bulyan);
                        default is distributed gradient descent without screening
```

## Examples

1) BRIDGE defending against at most two Byzantine nodes with no faulty nodes in the network (faultless setting).

```
$ python dec_BRIDGE.py 0 -b=2 -s=BRIDGE
```
2) BRIDGE defending against at most two Byzantine nodes with exactly two faulty nodes in the network (faulty setting).

```
$ python dec_BRIDGE.py 0 -b=2 -gb=True -s=BRIDGE
```

The user can run each of the possible screening methods ten times in parallel by varying `monte_trial` between 0 and 9 for ten independent Monte Carlo trials with predetermined random number generator seeds for each trial meant to reproduce the results in every run.

<a name="byrdie"></a>
# ByRDiE Experiments
We performed decentralized learning using ByRDiE, both in the faultless setting and in the presence of actual Byzantine nodes. To train the one layer neural network on MNIST with ByRDiE, run the `dec_ByRDiE.py` script. Each Monte Carlo trial for ByRDiE ran in about two days on our machines.

```
usage: dec_ByRDiE.py [-h] [-b BYZANTINE] [-gb GOBYZANTINE] monte_trial

positional arguments:
  monte_trial           A number between 0 and 9 to indicate which
  					  Monte Carlo trial to run

optional arguments:
  -h, --help            Show this help message and exit
  -b BYZANTINE, --byzantine BYZANTINE
                        Maximum number of Byzantine nodes to defend
                        against; if none then it defaults to 0
  -gb GOBYZANTINE, --goByzantine GOBYZANTINE
                        Boolean to indicate if the specified number of
                        Byzantine nodes actually send out faulty values
```

## Examples
1) ByRDiE defending against at most two Byzantine nodes with no faulty nodes in the network (faultless setting).

```
$ python dec_ByRDiE.py 0 -b=2
```

2) ByRDiE defending against at most two Byzantine nodes with exactly two faulty nodes in the network (faulty setting).

```
$ python dec_ByRDiE.py 0 -b=2 -gb=True
```

The user can run ByRDiE ten times in parallel by varying `monte_trial` between 0 and 9 for ten independent Monte Carlo trials with predetermined random number generator seeds for each trial meant to reproduce the results in every run.

<a name="plotting"></a>
# Plotting
All results generated by `dec_BRIDGE.py` and `dec_ByRDiE.py` get saved in `./result` folder. After running ten independent trials for each Byzantine-resilient decentralized method as described above, run the `plot.py` script to generate the plots similar to Figure 3 in the paper (Yang et al., 2020).

**Note:** Due to a loss in the original implementation of the decentralized Krum and Bulyan screening methods, the experiments with these screening methods will not perfectly reproduce the results found in Figure 3 of (Yang et al., 2020). Nonetheless, the results from the implementations in this codebase are consistent with the discussions and conclusions made in the paper.

# Contributors
The algorithmic implementations and experiments were originally developed by the authors of the papers listed above:

- [Zhixiong Yang](https://www.linkedin.com/in/zhixiong-yang-67139152/)
- [Arpita Gang](https://www.linkedin.com/in/arpita-gang-41444930/)
- [Waheed U. Bajwa](http://www.inspirelab.us/)

The reproducibility of this codebase and publicizing of it was made possible by:

- [Joseph Shenouda](https://github.com/joeshenouda)
