apt-get update 
apt-get install libopenmpi-dev
pip install numpy==1.18.5
rm -rf /workspace/spinningup
cd /workspace
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

