# Lander

```bash
# venv setup
python3.9 -m venv openaigym-venv
pip install -U pip setuptools wheel
pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu116
pip install ipython ipdb tensorboard
pip install "stable-baselines3[extra]==1.6.0" gym[box2d,atari] autorom[accept-rom-license]
AutoROM --accept-license
```

```
# setup.py is from python packaging project sample
https://raw.githubusercontent.com/pypa/sampleproject/main/setup.py
```

```
pip install --upgrade gym -f git+https://github.com/openai/gym
```
