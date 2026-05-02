curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_LINK_MODE=copy
source $HOME/.local/bin/env



apt-get update && apt-get install -y python3.10-dev build-essential


apt install -y unzip screen


# git config
git config --global user.email "joaquin.jordan@utec.edu.pe"
git config --global user.name "Joaquin Jordan"
git config --global --add safe.directory '*'
git config --global pull.rebase false