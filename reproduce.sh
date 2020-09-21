#!/bin/bash
# There are three ways to reproduce the results of the paper, along with
# the text and slides
# 
# 1. On ANY computer with these two items installed:
# 
#    [nbreproduce](https://github.com/econ-ark/nbreproduce)
#    [docker](https://en.wikipedia.org/wiki/Docker_(software))
#
# From the root directory of the Pandemic project, execute the command
# 
#     nbreproduce
#
# 2. On any unix computer, 
# First, git clone https://github.com/econ-ark/Pandemic
# Next, install the requirements using the command on the next line:
# python3 -m pip install -r requirements.txt
#
# and finally just execute this reproduce.sh script itself:
# /bin/bash reproduce.sh

# 3. Obtain the standard econ-ark docker image:
# https://github.com/econ-ark/econ-ark-tools/blob/master/Virtual/Docker/README.md
#
# and then, after launching the docker machine, follow the instructions for unix above

# Create all results
cd Code/Python
python GiveItAwayMAIN.py

# Back to root directory
cd ../..

# Compile ConsumptionResponse.tex
pdflatex -output-directory=LaTeX ConsumptionResponse.tex
bibtex LaTeX/ConsumptionResponse
pdflatex -output-directory=LaTeX ConsumptionResponse.tex
pdflatex -output-directory=LaTeX ConsumptionResponse.tex

# Compile ConsumptionResponse-Slides.tex
pdflatex -output-directory=LaTeX ConsumptionResponse-Slides.tex
bibtex LaTeX/ConsumptionResponse-Slides
pdflatex -output-directory=LaTeX ConsumptionResponse-Slides.tex
pdflatex -output-directory=LaTeX ConsumptionResponse-Slides.tex
