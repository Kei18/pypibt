# pypibt

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](./LICENCE.txt)
[![CI](https://github.com/Kei18/pypibt/actions/workflows/ci.yml/badge.svg)](https://github.com/Kei18/pypibt/actions/workflows/ci.yml)

A minimal python implementation of Priority Inheritance with Backtracking (PIBT) for Multi-Agent Path Finding (MAPF).
If you are just interested in moving hundreds of agents or more in a short period of time, PIBT may work as a powerful tool.

- Okumura, K., Machida, M., Défago, X., & Tamura, Y. Priority inheritance with backtracking for iterative multi-agent path finding. AIJ. 2022. [[project-page]](https://kei18.github.io/pibt2/)

## background

To be honest, as the developer of PIBT, I only developed it to keep multiple agents running smoothly, not to solve MAPF or MAPD.
But it turned out to be much more powerful than I expected.
A successful example is [LaCAM*](https://kei18.github.io/lacam2/).
It achieves remarkable performance, to say the least.
I also noticed that PIBT has been extended and used by other researchers.
These experiences were enough to motivate me to create a minimal implementation example to help other studies, including my future research projects.

As you know, many researchers like Python because it is friendly and has a nice ecosystem.
In contrast, most MAPF algorithms, such as the original PIBT, are coded in C++ for performance reasons.
So here is the Python implementation.
I hope the repo is helpful to understand the algorithm; the main part is only a hundred and a few lines.
You can also use and extend this repo, for example, applying to new problems, enhancing with machine learning, etc.

## setup

This repository is easily setup with [Poetry](https://python-poetry.org/).
After cloning this repo, run the following to complete the setup.

```sh
poetry install
```

## demo

```sh
poetry run python app.py -m assets/random-32-32-10.map -i assets/random-32-32-10-random-1.scen -N 200
```

The result will be saved in `output.txt`
The grid maps and scenarios in `assets/` are from [MAPF benchmarks](https://movingai.com/benchmarks/mapf/index.html).

### visualization

You can visualize the planning result with [@kei18/mapf-visualizer](https://github.com/kei18/mapf-visualizer).

```sh
mapf-visualizer ./assets/random-32-32-10.map ./output.txt
```

![](./assets/demo.gif)

### jupyter lab

Jupyter Lab is also available.
Use the following command:

```sh
poetry run jupyter lab
```

You can see an example in `notebooks/demo.ipynb`.


## Licence

This software is released under the MIT License, see [LICENSE.txt](LICENCE.txt).
