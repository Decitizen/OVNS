# OVNS: Opportunistic Variable Neighborhood Search for Heaviest Subgraph Problem in Social Networks

Welcome to the official GitHub repository for the [research paper (arxiv)](https://arxiv.org/abs/2305.19729): 

_"OVNS: Opportunistic Variable Neighborhood Search for Heaviest Subgraph Problem in Social Networks"_ by Ville P. Saarinen, Ted Hsuan Yun Chen, Mikko Kivelä.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Benchmarks](#results-and-benchmarks)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Getting Started

This repository contains the code base for the python implementation of the OVNS algorithm, a state-of-the-art performant heuristic for solving the Heaviest k-Subgraph Problem in social networks. 

### Prerequisites

This software has the following dependencies:

- Python (3.9 or newer)
- Numba (0.56 or newer)
- Numpy

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/decitizen/ovns.git
    ```

2. Install Python dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

After installation, you can run the OVNS algorithm on your own social network data.

```python
# Import ovns package
import ovns
import numpy as np

N = 1000                               # Number of nodes in the network 
rng = np.random.default_rng()
A = rng.standard_exponential((N, N))   # Random adjacency matrix carrying the weight information
k = 20                                 # Size of the targeted subgraph

result = OVNS(A, k)

```

## Results and Benchmarks

We have conducted benchmarks in both real-life social networks as well as synthetic networks. Please refer to our paper for a comprehensive explanation and discussion of these results.

## Contributing

We welcome contributions to the OVNS project. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

If you have any questions or comments, please feel free to reach out to the authors @decitizen via email ville.saarinen@tuta.io

