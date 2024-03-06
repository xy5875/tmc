#!/bin/bash

conda activate pytorch2

# python start_server.py --logname 异步_IID_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/new_iid_data --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_NIID0.1_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.1 --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_NIID0.01_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.01 --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_NIID0.2_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.2 --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_NIID0.3_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.3 --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_NIID0.4_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.4 --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_NIID0.5_不使用mask_不使用anchor_无时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.5 --delay False&

# sleep 7200

# pkill python

# python start_server.py --logname 异步_IID_不使用mask_不使用anchor_有时延 --port 8081&

# sleep 3

# python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/new_iid_data --delay True&

# sleep 7200

# pkill python

python start_server.py --logname 异步_NIID0.1_不使用mask_不使用anchor_有时延 --port 8081&

sleep 3

python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.1 --delay True&

sleep 7200

pkill python

python start_server.py --logname 异步_NIID0.01_不使用mask_不使用anchor_有时延 --port 8081&

sleep 3

python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.01 --delay True&

sleep 7200

pkill python

python start_server.py --logname 异步_NIID0.2_不使用mask_不使用anchor_有时延 --port 8081&

sleep 3

python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.2 --delay True&

sleep 7200

pkill python

python start_server.py --logname 异步_NIID0.3_不使用mask_不使用anchor_有时延 --port 8081&

sleep 3

python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.3 --delay True&

sleep 7200

pkill python

python start_server.py --logname 异步_NIID0.4_不使用mask_不使用anchor_有时延 --port 8081&

sleep 3

python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.4 --delay True&

sleep 7200

pkill python

python start_server.py --logname 异步_NIID0.5_不使用mask_不使用anchor_有时延 --port 8081&

sleep 3

python start_client.py --anchor False --port 8081 --mask False --dataroot /home/dell/xy/new/mobicom-TWT/cifar/train_data_0.5 --delay True&

sleep 7200

pkill python



# python start_server.py --logname 不使用Mask --port 8082 &

# sleep 3

# python start_client.py --anchor False --port 8082 --mask False&