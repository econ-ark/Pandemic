#!/bin/bash
# On a unix computer will all of the software requirements installed,
# excution of this script should reproduce all of the results of the paper

# On ANY computer with these two items installed:
# 
#    [nbreproduce](https://github.com/econ-ark/nbreproduce)
#    [docker](https://en.wikipedia.org/wiki/Docker_(software))
#
# All the results of the paper should be reproduced by invoking the command
# 
#     nbreproduce --docker econark/pandemic
#
# If using the standard econ-ark-notebook docker image,      % MS: Please add link to your documentation explaining this
# uncomment the following line to install the requirements. 
#
# python3 -m pip install -r requirements.txt

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
