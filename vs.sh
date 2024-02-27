#!/bin/bash
source /home/dell/miniconda3/bin/activate pytorch2


#MODEL--->LABEL
#A--->4(C)(2)
#B--->9(E)(4)
#C--->6(D)(3)
#D--->3(B)(1)
#E--->0(A)(0)
#AVG--->2(B)(1)


python vs.py --train_path AFLvsGFL/data/drift_ez/A --test_path AFLvsGFL/data/drift_ez/A --model_path /home/dell/xy/AFLvsGFL/model/ONE/A/A.pth >>A.txt&
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_4 --test_path /home/dell/xy/AFLvsGFL/data/test/4 --model_path /home/dell/xy/AFLvsGFL/model/ONE/B/B.pth >>B.txt&
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_3 --test_path /home/dell/xy/AFLvsGFL/data/test/3 --model_path /home/dell/xy/AFLvsGFL/model/ONE/C/C.pth >>C.txt&
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_1 --test_path /home/dell/xy/AFLvsGFL/data/test/1 --model_path /home/dell/xy/AFLvsGFL/model/ONE/D/D.pth >>D.txt&
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_0 --test_path /home/dell/xy/AFLvsGFL/data/test/0 --model_path /home/dell/xy/AFLvsGFL/model/ONE/E/E.pth >>E.txt&
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_2 --test_path /home/dell/xy/AFLvsGFL/data/test/1 --model_path /home/dell/xy/AFLvsGFL/model/gfl/AVG.pth >>AVG.txt&
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_0 --test_path /home/dell/xy/AFLvsGFL/data/test/0 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_3 --test_path /home/dell/xy/AFLvsGFL/data/test/3 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_4 --test_path /home/dell/xy/AFLvsGFL/data/test/4 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth


python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/A --test_path /home/dell/xy/AFLvsGFL/data/test_ez/A --model_path /home/dell/xy/AFLvsGFL/model/ONE/A/A.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/B --test_path /home/dell/xy/AFLvsGFL/data/test_ez/B --model_path /home/dell/xy/AFLvsGFL/model/ONE/B/B.pth 
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/C --test_path /home/dell/xy/AFLvsGFL/data/test_ez/C --model_path /home/dell/xy/AFLvsGFL/model/ONE/C/C.pth 
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/D --test_path /home/dell/xy/AFLvsGFL/data/test_ez/D --model_path /home/dell/xy/AFLvsGFL/model/ONE/D/D.pth 
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/E --test_path /home/dell/xy/AFLvsGFL/data/test_ez/E --model_path /home/dell/xy/AFLvsGFL/model/ONE/E/E.pth 


python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/A --test_path /home/dell/xy/AFLvsGFL/data/test_ez/A --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/B --test_path /home/dell/xy/AFLvsGFL/data/test_ez/B --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/C --test_path /home/dell/xy/AFLvsGFL/data/test_ez/C --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth 
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/D --test_path /home/dell/xy/AFLvsGFL/data/test_ez/D --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_ez/E --test_path /home/dell/xy/AFLvsGFL/data/test_ez/E --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth

python vs.py --train_path /home/dell/xy/AFLvsGFL/data/train_avg/3 --test_path /home/dell/xy/AFLvsGFL/data/test_avg/0 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth

python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_0 --test_path /home/dell/xy/AFLvsGFL/data/test/0 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_1 --test_path /home/dell/xy/AFLvsGFL/data/test/1 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_2 --test_path /home/dell/xy/AFLvsGFL/data/test/2 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_3 --test_path /home/dell/xy/AFLvsGFL/data/test/3 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth
python vs.py --train_path /home/dell/xy/AFLvsGFL/data/drift_4 --test_path /home/dell/xy/AFLvsGFL/data/test/4 --model_path /home/dell/xy/AFLvsGFL/model/pt/1.pth