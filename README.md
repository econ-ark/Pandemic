# Pandemic-ConsumptionResponse
 
This repository is a complete software archive for the paper "Modeling the Consumption
Response to the CARES Act" by Carroll, Crawley, Slacalek, and White (2020).  This README file
provides instructions for running our code on your own computer, as well as adjusting the parameters of the model to produce alternate versions of the figures in the paper.

## REPRODUCTION INSTRUCTIONS

0. All of the code for the project is written in Python 3, and is intended to be run in an
iPython graphical environment.  Running the main script outside of iPython may cause unintended
consequences when figures are created.

1. The easiest way to get iPython running on your computer is to use the Anaconda distribution
of Python, available for download at https://www.anaconda.com/distribution/

2. The code for this project uses the [Heterogeneous Agents Resources and toolKit](http://github.com/econ-ark/HARK) to solve and simulate our model.  To install HARK, open a console (on Windows, use the Anaconda Prompt) and run `pip install econ-ark`.  This will put HARK and all of its dependencies in your Python package library.

3. All code files are in the `./Code/Python` subdirectory of this repository.  If you've installed
Anaconda, our code can be run in a graphical iPython environment by opening up the Spyder IDE.

4. The main script for this project is GiveItAwayNowMAIN.py.  You can run this file by clicking
the green arrow "run" button in Spyder's toolbar.  Text will print to screen as it completes
various computational tasks.  Most of these take about 1-2 minutes to run on a modern desktop.
All figures are produced after running all counterfactual scenarios.

5. All figures are saved to the ./Figures subdirectory.

6. All parameters can be adjusted in ./Python/parameter_config.py , and are described below.
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

4. The third parameter block specifies the terms of the fiscal stimulus package (CARES Act), including the timing of the stimulus checks relative to announcement, the size of the stimulus checks,
the term of the income-based means testing, the size of additional unemployment benefits, and the
proportion of the population who notices and reacts to the announement of the stimulus before the
checks actually arrive in their bank account.  Note that all values are specified in thousands of
dollars, and the model is quarterly.

5. The fourth parameter block includes basic model parameters like the population growth rate,
aggregate productivity growth rate, and a description of "normal" unemployment (and benefits).

6. The fifth parameter block specifies the initial distribution of permanent income for each education level and the share of the education levels in the population.

7. The remaining parameters specify the density of the grid that the problem is solved on and the
discretization of the income shocks (for computing expectations).

8. Most parameters below the line of pound signs should not be adjusted, as they concern objects
constructed from the primitive parameters defined above or basic features of the lifecycle (such
as the number of periods in the problem).  The number of periods simulated in the counterfactuals
and the total number of simulated households can be safely changed.
