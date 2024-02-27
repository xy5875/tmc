Build base on flask and torch
You need to modify ```CUDA_LIST``` and ```RAW_CUDA_LIST``` to satisfy with your cuda device
## run server
```python start_server.py```
## run client
```python start_client.py```
start_client is employed to run 120 clients parallelly.
To run one client ```python client_wapper.py```
## log file
You can find the run log at ```log_files/server_log.txt```