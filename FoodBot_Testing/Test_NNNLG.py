import math
import os
import sys
import time
import json
import argparse

sys.path.append('RNNLG/')
from generator.net import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default RNNLG opt parser.')

    parser.add_argument('-mode',  help='modes: train|test|adapt|knn|ngram', default="test")
    parser.add_argument('-config', help='config file to set.',default="RNNLG/config/sclstm.cfg")
    args = parser.parse_args()
    print (args.config,args.mode)
    model = Model(args.config,args)
    print (model.testNet())
