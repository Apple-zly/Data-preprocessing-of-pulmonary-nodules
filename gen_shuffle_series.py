import numpy as np
import math
import os

from utils import readCsv, writeCsv

from random import shuffle

import pdb

data_ip = '/home/data/zly/LungnoduleDetection/LNDb/prep_seg_1800_all/'
filenames = os.listdir(data_ip)
series = []
for filename in filenames:
    if '_clean' in filename:
        # print(filename)
        name = str(filename.split('_')[0])
        print(name)
        series.append([name])

pdb.set_trace()
shuffle(series)

writeCsv('serieses.csv', series)