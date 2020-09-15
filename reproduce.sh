# This can be removed if Pandemic specific docker image is used
# nbreproduce --docker econark/Pandemic
python3 -m pip install -U -r requirements.txt

# recreate all figures
cd Code/Python
python GiveItAwayMAIN.py

# Add LaTeX compilation?
