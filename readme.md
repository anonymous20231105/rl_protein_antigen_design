# Installation
## install cuda11.3
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
```

```bash
sudo sh cuda_11.3.0_465.19.01_linux.run
```

## install openfold
see https://github.com/aqlaboratory/openfold/

## install requirements
```bash
pip install -r requirements.txt --default-timeout=1000
```

# Run
```bash
python ./src/ppo_train.py
```

# Test
```bash
python ./src/ppo_test.py
```
