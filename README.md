# BTC Predict

This is a Bitcoin bull run prediction project which aims to evaluate current bull run's peak price alongside the exact date. The project consists of two individual machine-learning models trained on the Bitcoin's historical pricing and block halving data.

Check out this awesome LIVE demo website: [btcpredict.monicz.pl](https://btcpredict.monicz.pl/)

## Usage

*Recommended Python version: 3.7*

### Install required packages

`$ pip install -r requirements.txt`

### Generate a prediction

`$ py main.py --file <filename> [--force_cache_update]`

* Create a `cache` directory and ensure it is writtable
* Force the cache update at least once a day

## Data transformation

The current prediction models are using the following data transformation methods:

| Name | Link |
|---------------|------|
| 2-Year MA Multiplier | [Visit page](https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/) |
| Pi Cycle Top Indicator | [Visit page](https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/) |
| Golden 51%-49% Ratio | [Visit page](https://www.tradingview.com/chart/BTCUSD/QBeNL8jt-BITCOIN-The-Golden-51-49-Ratio-600-days-of-Bull-Market-left/) |
| The Puell Multiple | [Visit page](https://www.lookintobitcoin.com/charts/puell-multiple/) |

*Inspired by the [CBBI](https://www.youtube.com/watch?v=ZFQG59ZMSU0).*

## Footer

### Contact

* Email: [kamil@monicz.pl](mailto:kamil@monicz.pl)
* PGP: [0x9D7BC5B97BB0A707](https://gist.github.com/Zaczero/158da01bfd5b6d236f2b8ceb62dd9698)

### Donate

Please consider supporting this project if you found it useful.
Any kind of donation will encourage me to focus on its further development.
You can send me an email afterwards to get listed in the supporters section *(not yet present)*.

#### Bitcoin

`bc1ql3gx9swg5zsn8ax8w34jw85juc5nqtprcdxrje`

#### Bitcoin Cash

`qrvqsfz2vj6p0zdpg7w7zfah7qag2ygpju7yqh05hu`

#### Litecoin

`ltc1qpjz5rhaas0lxf90re0u65sy5jujxhtuqwwerkd`

#### Monero

`4ABpPJchKYS8Nq9dPb8mr1NEEAHbKDr5aK777QZh2aSD7BJHdhkQn4RFQ3zNW2kytSXHXpimt57L9X9iin3uJjw93pCpKaJ`
