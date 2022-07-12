import argparse
from utils import backupfile


parser = argparse.ArgumentParser()
parser.add_argument('--t', '-t', default='default', type=str, required=False)
parser.add_argument('--log', default='./log')

args = parser.parse_args()

backupfile(args=args)

 
