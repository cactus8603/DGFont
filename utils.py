import argparse
from shutil import copyfile
from glob import glob
import os
from datetime import datetime

def backupfile(args):
    if not os.path.isdir(args.log):
        os.mkdir("log")

    timeinfo = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.isdir(os.path.join(args.log, timeinfo)):
        os.mkdir(os.path.join(args.log, timeinfo))

    pyfiles = glob("./*.py")
    for py in pyfiles:
        copyfile(py, os.path.join(args.log, timeinfo, py))

    dirs = next(os.walk('./'))[1]
    no_dirs = ['.git', 'log']
    for no_dir in no_dirs:
        dirs.remove(no_dir)

    for dir in dirs:
        if not os.path.exists(os.path.join(args.log, dir)):
            os.mkdir(os.path.join(args.log, timeinfo, dir))
        
        pyfiles = glob(os.path.join('./', dir, '*.py'))
        for py in pyfiles:
            copyfile(py, os.path.join(args.log, timeinfo, dir, os.path.split(py)[1]))