## Setup
### Rust installation
Rust is needed for compiling some parts of `outlines` module.
```shell
curl https://sh.rustup.rs -sSf | sh
```
Make rust available in current shell instance
```shell
. "$HOME/.cargo/env"  
```
Note: this behaves strangely if you have Rust already installed. I recommend reinstalling it (I had problems with Rust installed via `apt`).
### Dependencies install
```shell
pip install -r research/generativeLlms/requirements.txt
```
or whether your working dir is