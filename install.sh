apt-get update

git config --global user.email "woodychang0611@gmail.com"
git config --global user.name "Woody Chang"
pip install numpy==1.18.5
pip install  --upgrade  tqdm
apt-get install libopenmpi-dev

rm -rf /workspace/spinningup
cd /workspace
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

