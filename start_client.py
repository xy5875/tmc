import subprocess
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--anchor')
parser.add_argument('--port')
parser.add_argument('--mask')
parser.add_argument('--dataroot')
parser.add_argument('--delay')
parser.add_argument('--seed')
parser.add_argument('--testfile',help = "test file")
args = parser.parse_args()
import time

if __name__ == '__main__':
    names = [0,1,2,3,4,5,6,7,8,9]
    delay = [3,3,3,3,3,0,0,0,0,0]
    if args.delay == 'True':
        delay = [i*1 for i in delay]
    else:
        delay = [i*0 for i in delay]
    for index,name in enumerate(names):
        #print("name:",name)
        seed = int(time.time())
        cmd = f'python client_wapper.py --name {name} --anchor {args.anchor} --port {args.port} --mask {args.mask} --seed {seed} --dataroot {args.dataroot} --delay {delay[index]} --testfile {args.testfile}'
        cmd = cmd.split(' ')
        subprocess.Popen(cmd, shell=False)
        time.sleep(0.1)