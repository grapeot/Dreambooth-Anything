# shell environment...
sudo apt-get update
sudo apt-get install -y -q vim zsh git wget dos2unix python3 python3-pip parallel tig build-essential curl htop rsync tmux zip unzip pkg-config
wget --no-check-certificate https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | sh
pushd ~
git clone https://github.com/rupa/z
popd
# .dotfiles
pushd ~
git clone --recursive https://github.com/grapeot/.dotfiles
./.dotfiles/deploy_linux.sh
popd
# git configuration
git config --global user.name "Yan Wang"
git config --global user.email grapeot@outlook.com
git config --global push.default simple # eliminate the warning message of the new version git
git config --global color.ui auto
git config --global core.fileMode false
sudo pip3 install trash-cli virtualenv

wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
chmod +x cuda_11.6.0_510.39.01_linux.run
sudo ./cuda_11.6.0_510.39.01_linux.run --toolkit --silent

# python specific
cd ~
virtualenv -p python3.10 py310
source ~/py310/bin/activate
echo 'source ~/py310/bin/activate' >> ~/.zshrc
export PATH=$PATH:/usr/local/cuda/bin
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"' >> ~/.zshrc
cd Dreambooth-Anything
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
cd ..
git clone https://github.com/TimDettmers/bitsandbytes
cd bitsandbytes
wget https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-11.1.0/libstdc%2B%2B-v3/include/bits/std_function.h
sudo mv std_function.h /usr/include/c++/11/bits/
CUDA_VERSION=116 make cuda11x
python setup.py install

chsh -s $(which zsh)
