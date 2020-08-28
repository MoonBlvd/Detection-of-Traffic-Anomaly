import yaml
import argparse

def visualize_config(args):
    """
    Visualize the configuration on the terminal to check the state
    :param args:
    :return:
    """
    print("\nUsing this arguments check it\n")
    for key, value in sorted(vars(args).items()):
        if value is not None:
            print("{} -- {} --".format(key, value))
    print("\n\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', 
                        dest='config_file',
                        help='The yaml configuration file')
    args, unprocessed_args = parser.parse_known_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.load(f))
    
    args = parser.parse_args(unprocessed_args)
    visualize_config(args)
    return args
