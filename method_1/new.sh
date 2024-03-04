#!/bin/bash

conda activate pytorch2

python start_server.py --logname afl_AVG --port 8081 --client 0 1 2 3  --parallelnum 4 --testfile /home/dell/xy/AFLvsGFL/data/test/2 --flie /home/dell/xy/AFLvsGFL/model/gfl&

sleep 10

python start_client.py --anchor False --port 8081 --mask False  --delay True --dataroot /home/dell/xy/Fisher/data/C --testfile /home/dell/xy/AFLvsGFL/data/test/2 --taile True


sleep 14400

pkill python