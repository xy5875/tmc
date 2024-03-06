from time import sleep
from flask import Flask
from server import server_app
import threading
import requests
from server import init_server
import server_global_variable
from logging.config import dictConfig
import logging

def set_dictConfig(log_name):
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] : %(message)s',
        }},
        'handlers': {
            'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        },
            'test':{
                'class': 'logging.FileHandler',
                'filename':f"./log_files/{log_name}.txt",
                'formatter':'default',
            },
                },
        'root': {
            'level': 'WARNING',
            'handlers': ['wsgi','test']
        }
        })

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(21)


app.register_blueprint(server_app)

@app.route("/")
def register():
    return "hello world"

def run_app(port):
    app.run(port=port,host="0.0.0.0")
    
import torch
import os
from server_global_variable import Server_Status
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logname',default='default',help='log file name')
parser.add_argument('--info',default='data/info.json',help='info path')
parser.add_argument("--parallelnum",default=2,help='parallel number of training clients')
parser.add_argument('--port',default=8080)
parser.add_argument('--flie',help = "model save road")
parser.add_argument('--client', nargs='+', help="model save road")
parser.add_argument('--testfile',help = "test file")


from server import test

def exp_test_wapper(server_status):
    while(True):
        sleep(3)
        #test(server_status)
        print("---------------waiting next client----------------------\n")
    

if __name__=="__main__":
    server_status = Server_Status()
    args = parser.parse_args()
    init_server(args)
    set_dictConfig(args.logname)
    t1 = threading.Thread(target=exp_test_wapper,args=(server_status,))
    t1.start()
    with torch.no_grad():
        run_app(args.port)
    
    