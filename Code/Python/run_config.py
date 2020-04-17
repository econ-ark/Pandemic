import argparse
import yaml
from GiveItAwayVoila import run_web

# Take the file as an argument
parser = argparse.ArgumentParser()
parser.add_argument("config", help="A YAML config file for custom parameters")
args = parser.parse_args()
mystr = lambda x: "{:.2f}".format(x)

with open(args.config, "r") as stream:
    config_parameters = yaml.safe_load(stream)

print(config_parameters)

def run_configurator(config_parameters):
    for parameter, params in config_parameters.items():
        new_parameters = dict()
        append_list = list()
        for key, val in config_parameters[parameter].items():
            new_parameters[key] = val
            append_list.append(f'{key} = {val} \n')

        # write new parameters at the end of file
        with open('parameter_config.py', 'a') as f:
            for p in append_list:
                print(p)
                f.write(p)
        
        # create figures for the parameter
        run_web(parameter)
        
        # delete the new parameters and prepare it for the next run
        with open('parameter_config.py', 'rb') as f:
            d = f.readlines()
        with open('parameter_config.py', 'wb') as f:
            for i in range(len(d) - len(append_list)):
                f.write(d[i])

run_configurator(config_parameters)
