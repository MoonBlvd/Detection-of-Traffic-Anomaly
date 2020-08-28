'''
This is an example for generating dataset object for FOL model training and evalutaion.
'''
from dataset.dota import DoTADataset
from config.config import parse_args
import pdb

args = parse_args()

# initialize the dataset object used to train the model
dataset = DoTADataset(args, phase='train')

