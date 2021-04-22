# Urban driving games (UDGs) :red_car: :blue_car:

This repository provides the code associated to the following publication:

```
@article{zanardi2021udgs,
    title = {Urban Driving Games with Lexicographic Preferences and Socially Efficient Nash Equilibria},
    year = {2021},
    journal = {IEEE Robotics and Automation Letters},
    author = {Alessandro Zanardi; Enrico Mion; Mattia Bruschetta; Saverio Bolognani; Andrea Censi; Emilio Frazzoli},
    number = {-},
    pages = {-},
    volume = {-},
    url = {http://ieeexplore.ieee.org/document/4444573/},
    doi = {10.1109/LRA.2021.3068657},
    keywords = {Autonomous Agents, Motion and Path Planning, Optimization and Optimal Control, Game Theory}
}
```

The original code was in matlab, we ported it to python for future works. Minor differences might be present (mainly
plotting and visualisation, nothing substantial).

The solver used for solving the optimization problems
is [FORCES Pro](https://www.embotech.com/products/forcespro/overview/). Since it is a proprietary solver, to run the
experiments yourself there are only two options:

1. (download and use a docker image with precompiled solvers, provided at #TODO;)
2. contact Embotech and ask for a license (it might be free of charges if you are in academia).

With option (1) you can tweak some parameters but not much as any modification that affects the defined model would
require to recompile the solvers. With option (2) you are free to change the code as you wish, define new scenarios and
test them.

Either way we provide some interactive experiments reports at #todo

### Setting up the package in Pycharm

Make `forcespro` available to the interpreter:
Settings -> Project:..->Python Interpreter->Click on the wheel->Show all -> Show paths for the selected interpreter (the
one for teh project)-> add path to forcespro folder

### How to...

The `main.py` is the only "entrypoint" of the repository, from there following the code should be pretty straight
forward.

### Experiments

As getting the solver license might require some time, we provide some sampled experiments
in [this google dive folder](https://drive.google.com/drive/folders/189rVGHaBk5ja5GIIK9COg8HtWiy5GbAf?usp=sharing).
