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
                        # type=argparse.FileType(mode='r'),
                        help='The yaml configuration file')
    args, unprocessed_args = parser.parse_known_args()

    # parser.add_argument('--data_root', default=None, required=True, help='The data folder')
    # parser.add_argument('--phase', default=None, required=True, help='train or val')

    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.load(f))
    
    args = parser.parse_args(unprocessed_args)
    visualize_config(args)
    return args
