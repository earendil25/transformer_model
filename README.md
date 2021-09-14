# transformer_model
Transformer based prediction model ensembled with resnet model.
## Feature
We use 60 monthly data of 54 economic variables from FRED and QUANDL. Companies that have entered top 500 in US stock market are collected.
<br>
![feature](https://user-images.githubusercontent.com/73049948/133258875-519a9b93-c186-426d-97a8-8a3e04642d4b.png)

## Label
Binary boolean label. True = ticker outperforms S&P500TR in the following month.

## Model structure
<img src="https://user-images.githubusercontent.com/73049948/133261434-4c6717c0-585d-4f78-9153-756086083f7d.PNG" width="300" >


## Backtest result
Benchmark = S&P500TR
### Cap-weighted
![CW](https://user-images.githubusercontent.com/73049948/133259349-2ab092ab-ca45-4a10-b8e9-f4bdc3347fe8.png)
### Factor-weighted
![FW](https://user-images.githubusercontent.com/73049948/133259412-be06d573-d4e1-499d-9043-af53249b5ba2.png)
