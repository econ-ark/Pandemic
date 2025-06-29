# Pandemic-ConsumptionResponse
 
This repository is a complete software archive for the paper ["Modeling the Consumption
Response to the CARES Act"](https://econ-ark.github.io/Pandemic) by Carroll, Crawley, Slacalek, and White (2020).  This README file
provides instructions for running our code on your own computer, as well as adjusting the
parameters of the model to produce alternate versions of the figures in the paper.

## Reproduce using nbreproduce.

This repository and the [paper](https://econ-ark.github.io/Pandemic) can be reproduced using [nbreproduce](https://econ-ark.github.io/nbreproduce/#installation) and the corresponding [docker image](https://hub.docker.com/repository/docker/econark/pandemic) of this repository.

```
$ nbreproduce --docker econark/pandemic
```

## Running the Dashboard locally.

You can also run the dashboard locally using [pixi](https://pixi.sh/). If you do not have pixi installed on your machine you can follow the [installation instructions](https://pixi.sh/latest/installation/). Navigate to the `Pandemic` folder locally and run the following command:

```
$ pixi run dashboard
```

This should start up a local server for the dashboard and open up the dashboard in your default browser. Use `Ctrl + c` to stop the process in your terminal.

## REPRODUCTION INSTRUCTIONS in your local environment.

0. All of the code for the project is written in Python 3, and is intended to be run in an
iPython graphical environment.  Running the main script outside of iPython may cause unintended
consequences when figures are created.

1. The easiest way to get iPython running on your computer is to use the Anaconda distribution
of Python, available for download at https://www.anaconda.com/distribution/

2. The code for this project uses the [Heterogeneous Agents Resources and toolKit](http://github.com/econ-ark/HARK)
to solve and simulate our model.  To install HARK, open a console (on Windows, use the Anaconda
Prompt) and run `pip install econ-ark==0.10.7`.  This will put HARK and all of its dependencies in
your Python package library. HARK is still under development, so we strongly recommend you use this
exact version in order to ensure the code runs properly.

3. All code files are in the `./Code/Python/` subdirectory of this repository.  If you've installed
Anaconda, our code can be run in a graphical iPython environment by opening up the Spyder IDE.

4. The main script for this project is GiveItAwayNowMAIN.py.  You can run this file by clicking
the green arrow "run" button in Spyder's toolbar.  Text will print to screen as it completes
various computational tasks.  Most of these take about 3 minutes to run on a modern desktop,
but there are many of them.  The figures are produced after running all counterfactual scenarios,
and the entire run time is about 75 minutes.  Our main results hold when many fewer simulated
agents are used (say, 50,000 versus the 1,000,000 used in the code).

5. We recommend that you instead run the script GiveItAwayNowMINI.py, which produces a smaller
number of figures (and thus runs a smaller number of counterfactuals), saving the results to
a subdirectory of `./Figures/` given by spec_name at the top of `parameter_config.py`.

6. All figures are saved to the ./Figures subdirectory.

7. All parameters can be adjusted in ./Code/Python/parameter_config.py, and are described below.
Each parameter should have an in-line description to give you a pretty good sense of what it does.


## STRUCTURAL PARAMETERS

1. The parameters for the project are defined in parameter_config.py and imported en masse into
Parameters.py; they are divided into several blocks or types of parameters.

2. The distribution of the intertemporal discount factor (beta) is defined by the parameters in
the first block.  For each education level, we specify the center of a uniform distribution; the
half-width of the distribution is the same for all education levels (DiscFacSpread).

3. The second block of parameters is the largest, concerning what happens when the pandemic
strikes.  This includes the marginal utility scaling factor during the "lockdown", the real and
perceived duration of deep unemployment spells and the "lockdown", and coefficients for the logit
probabilities of employment and deep employment (see paper appendix).  It also includes a boolean
to indicate whether the lifting of the lockdown should be simulated as an idiosyncratic event
(as in the paper, to synthesize an *average* consumption path) or as a common event shared across
all agents (to see what happens with a *particular* timing of the return to normalcy).

4. The third parameter block specifies the terms of the fiscal stimulus package (CARES Act),
including the timing of the stimulus checks relative to announcement, the size of the stimulus checks,
the term of the income-based means testing, the size of additional unemployment benefits, and the
proportion of the population who notices and reacts to the announement of the stimulus before the
checks actually arrive in their bank account.  Note that all values are specified in thousands of
dollars, and the model is quarterly.

5. The fourth parameter block includes basic model parameters like the population growth rate,
aggregate productivity growth rate, and a description of "normal" unemployment (and benefits).

6. The fifth parameter block specifies the initial distribution of permanent income for each education
level and the share of the education levels in the population.

7. The remaining parameters specify the density of the grid that the problem is solved on and the
discretization of the income shocks (for computing expectations).

8. Most parameters in `Parameters.py` should not be adjusted, as they concern objects
constructed from the primitive parameters defined above or basic features of the lifecycle (such
as the number of periods in the problem).  The number of periods simulated in the counterfactuals
and the total number of simulated households can be safely changed.

9. Because of rounding, the actual number of simulated agents might be slightly different than the
number of agents specified in Parameters.py.  Agents are heterogeneous in their education level and
intertemporal discount factor; the fraction of each type times the total number of agents is unlikely
to result in a whole number, so the result is rounded. Summed across types, these rounded values do
not necessarily sum to the requested number of agents.
