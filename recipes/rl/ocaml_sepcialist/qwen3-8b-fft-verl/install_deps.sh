set -eux

git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
cd ..
apt-get install ocaml -y