# obtext

## Installation

See `install.sh`

## Usage

```
# ... run install.sh ...
conda activate obtext_env

cat test.txt | python -m obtext.bert > test.enc

# test.env is a 75x768 matrix
#   75  -> number of tokens in input
#   768 -> dimensionality of embedding
```
