import requests
import subprocess
import pickle
import time

def req_train(serverIp,serverPort,args):
    url = f'http://{serverIp}:{serverPort}/server/req_train'
    resp = pickle.loads(requests.post(url,data=pickle.dumps(args.name)).content)
    if resp['train']:
        cuda = resp['cuda']
        cmd = f'python client.py --name {args.name} --cuda {cuda} --anchor {args.anchor} --serverport {serverPort} --mask {args.mask} --seed {args.seed} --dataroot {args.dataroot} --delay {args.delay} --testfile {args.testfile}'
        cmd = cmd.split(' ')
        subprocess.run(cmd, shell=False)
    time.sleep(0.1)


def url_build_core(serverIp,serverPort):
    return f'http://{serverIp}:{serverPort}/server'

    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--anchor',default=True)
parser.add_argument('--port')
parser.add_argument('--mask')
parser.add_argument('--seed')
parser.add_argument('--dataroot')
parser.add_argument('--delay')
parser.add_argument('--testfile',help = "test file")

if __name__=='__main__':
    args = parser.parse_args()
    while(True):
        req_train('127.0.0.1',str(args.port),args)
    
    

