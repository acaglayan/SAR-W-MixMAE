# INSTALL.md

## 1) Environment
```bash
conda create -n sarwmix python=3.12 -y
conda activate sarwmix
```

## 2) PyTorch (CUDA 12.x)
Install the wheel that matches your CUDA driver:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python - <<'PY'
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

## 3) Dependencies
```bash
pip install -r requirements.txt
```

## 4) Quick check
```bash
python - <<'PY'
from sarwmix.bigearthnetv2 import BigEarthNetv2
print("sarwmix imports OK")
PY
```
