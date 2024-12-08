# Experiments for report, test influence of certain hyperparameters

import os

# 1. Test reproducibility, run twice the same experiment
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=5') #2024-11-14_10-25
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=5') #2024-11-14_10-37

# 2. Change init seed but same folds
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10') #2024-11-14_11-25 #seed 1234, for all below, the seed for split is 1234
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=0 --seed=2024') # Useless experiment, just to create data
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --seed=42 --no-create_data')
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --seed=2024 --no-create_data')
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --seed=4503 --no-create_data')

# 3. Reject outliers
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --flat_threshold=3.63 --reject_threshold=6.14')
# More lenien
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --flat_threshold=1. --reject_threshold=6.14')

# 4. Representation dim
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --rep_size=8') #2024-11-15_08-08
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --rep_size=16') #2024-11-14_13-36
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --rep_size=32') #2024-11-14_13-53
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --rep_size=64') #2024-11-14_14-11
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --rep_size=320') #2024-11-14_14-29

# 5. Model's size
# Base model: 149,315 parameters
# /2 108,131
# /4 88,019
# /8 78,083
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --dim_ff=128 --emb_size=8 --num_heads=4') #2024-11-14_15-08 #All transformer's params /2
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --dim_ff=64 --emb_size=4 --num_heads=2') #2024-11-14_15-30 #All transformer /4
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --dim_ff=32 --emb_size=2 --num_heads=1') #2024-11-15_08-26

# 6. Same params but different fold
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=0 --seed=2024') # Useless experiment, just to create data
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --seed=1234 --no-create_data') # split with seed 2024 but model initialised 1234
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=0 --seed=42') # Useless experiment, just to create data
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --seed=1234 --no-create_data') # split with seed 42 but model initialised 1234

# 7. Overlap
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --overlap_ratio=0.5') #2024-11-15_08-55

# 8. Small model, rep size = 2
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=10 --dim_ff=32 --emb_size=2 --num_heads=1 --rep_size=2') #2024-11-15_09-57

# 9. duration of signal
#os.system('/opt/conda/envs/py38/bin/python main.py --epochs=5 --dim_ff=128 --emb_size=8 --num_heads=4 --duration=4') #2024-11-15_14-46

# 10. Subsampling frequency
os.system('/opt/conda/envs/py38/bin/python main.py --epochs=5 --dim_ff=128 --emb_size=8 --num_heads=4 --sample_rate=50') #2024-11-15_15-07