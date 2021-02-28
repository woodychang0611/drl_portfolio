apt-get update
pip install numpy==1.18.5
pip install  --upgrade  tqdm
apt-get install libopenmpi-dev

rm -rf /workspace/spinningup
cd /workspace
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

