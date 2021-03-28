# BTC Predict

This is a Bitcoin bull run prediction project which aims to evaluate current bull run's top price alongside the exact date. The project is based of 2 individual ML models trained on the Bitcoin historical pricing and block halving data.

Check out this LIVE demo website: [btcpredict.monicz.pl](https://btcpredict.monicz.pl/)

## Usage

*Recommended Python version: 3.7*

### Install required packages

`$ pip install -r requirements.txt`

### Generate a prediction

`$ py main.py --file <filename> [--force_cache_update]`

* Create `cache` directory and ensure it is writtable
* Force cache update at least once a day

## Data transformation

The current prediction models are using the following data transformation methods:

| Name | Link |
|---------------|------|
| 2-Year MA Multiplier | [Visit page](https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/) |
| Pi Cycle Top Indicator | [Visit page](https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/) |
| Golden 51%-49% Ratio | [Visit page](https://www.tradingview.com/chart/BTCUSD/QBeNL8jt-BITCOIN-The-Golden-51-49-Ratio-600-days-of-Bull-Market-left/) |
| The Puell Multiple | [Visit page](https://www.lookintobitcoin.com/charts/puell-multiple/) |

*Inspired by the [CBBI](https://www.youtube.com/watch?v=rbkLLq5lVTA)*

## Footer

### Contact

* Email: [kamil@monicz.pl](mailto:kamil@monicz.pl)
* PGP: [0x9D7BC5B97BB0A707](https://gist.github.com/Zaczero/158da01bfd5b6d236f2b8ceb62dd9698)

### Donate

Please consider supporting this project if you found it useful.
Any kind of donation will encourage me to focus on its further development.
You can send me an email afterwards to get listed in the supporters section *(not yet present)*.

#### Bitcoin

`bc1qvhdxpwzcvfhyzlf5jd3xvm48tm5wm6d0xaq55z`

#### Bitcoin Cash

`qpclmw8ulcdqd2gd66ryepshn9dq8278tvmkpu6df6`

#### Monero

`4ABpPJchKYS8Nq9dPb8mr1NEEAHbKDr5aK777QZh2aSD7BJHdhkQn4RFQ3zNW2kytSXHXpimt57L9X9iin3uJjw93pCpKaJ`
