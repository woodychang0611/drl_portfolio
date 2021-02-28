apt-get update 
apt-get remove tensorflow
apt-get install libopenmpi-dev
rm -rf /workspace/spinningup
cd /workspace
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

