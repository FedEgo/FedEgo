seed=(1027 1107 2333 9973)
if [ ! -d ./result/out_$1 ]; then
  mkdir result/out_$1
fi
if [ $2 = -s ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=dfedgnn --seed=$3> result/out_$1/out_$1_dfedgnn_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedavg --seed=$3> result/out_$1/out_$1_fedavg_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3> result/out_$1/out_$1_fedego_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedgcn --seed=$3> result/out_$1/out_$1_fedgcn_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedprox --seed=$3> result/out_$1/out_$1_fedprox_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=graphfl --seed=$3> result/out_$1/out_$1_graphfl_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=local --seed=$3> result/out_$1/out_$1_local_$3.txt 2>&1 &
fi 

if [ $2 = --adaptive ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.125 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_00_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.25 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_01_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.375 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_02_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.5 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_03_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.625 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_04_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.75 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_05_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.875 --lamb_fixed=0 > result/lamb_$1/lamb_adaptive_06_$3.txt 2>&1 &
fi 
if [ $2 = --fixed ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.125 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_00_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.25 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_01_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.375 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_02_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.5 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_03_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.625 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_04_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.75 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_05_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --lamb_c=0.875 --lamb_fixed=1 > result/lamb_$1/lamb_fixed_06_$3.txt 2>&1 &
fi 

# major rate
if [ $2 = --major ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --major_rate=0.0 > result/major_$1/fedego_0_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --major_rate=0.3 > result/major_$1/fedego_3_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --major_rate=0.5 > result/major_$1/fedego_5_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedavg --seed=$3 --major_rate=0.0 > result/major_$1/fedavg_0_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedavg --seed=$3 --major_rate=0.3 > result/major_$1/fedavg_3_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedavg --seed=$3 --major_rate=0.5 > result/major_$1/fedavg_5_$3.txt 2>&1 &
fi 

# mixup 
if [ $2 = --mixup ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --mixup=0 --early_stopping=500 > result/mixup_$1/mixup_$3.txt 2>&1 &
fi 

# reduction & personalized 
if [ $2 = --abation ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedavg --seed=$3 --early_stopping=500 > result/abation_$1/fedavg_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --early_stopping=500 > result/abation_$1/fedego_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego_np --seed=$3 --early_stopping=500 > result/abation_$1/fedego_np_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego_nr --seed=$3 --early_stopping=500 > result/abation_$1/fedego_nr_$3.txt 2>&1 &
fi 

if [ $2 = --clients ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --client_num=5 --early_stopping=500 --timing=1 > result/clients_$1/5_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --client_num=15 --early_stopping=500 --timing=1 > result/clients_$1/15_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --client_num=20 --early_stopping=500 --timing=1 > result/clients_$1/20_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --client_num=25 --early_stopping=500 --timing=1 > result/clients_$1/25_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --client_num=30 --early_stopping=500 --timing=1 > result/clients_$1/30_$3.txt 2>&1 &
nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --client_num=35 --early_stopping=500 --timing=1 > result/clients_$1/35_$3.txt 2>&1 &
fi 

if [ $2 = --batch ]; then
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --batch_size=128 --early_stopping=500 --timing=1 > result/batch_$1/128_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --batch_size=16 --early_stopping=500 --timing=1 > result/batch_$1/16_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --batch_size=32 --early_stopping=500 --timing=1 > result/batch_$1/32_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --batch_size=64 --early_stopping=500 --timing=1 > result/batch_$1/64_$3.txt 2>&1 &
  nohup python3 main.py --cuda --dataSet=$1 --mode=fedego --seed=$3 --batch_size=8 --early_stopping=500 --timing=1 > result/batch_$1/8_$3.txt 2>&1 &
fi 