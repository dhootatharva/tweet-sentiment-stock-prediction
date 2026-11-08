# Results

## Replication of the original setup (test R² on closing price)

|      |   SVR, original scaling |   Random forest, original scaling |   SVR, train-only scaling |   Random forest, train-only scaling |   Yesterday's close (no model) |
|:-----|------------------------:|----------------------------------:|--------------------------:|------------------------------------:|-------------------------------:|
| AAPL |                   0.891 |                             0.866 |                    -2.074 |                              -0.796 |                          0.991 |
| AMZN |                   0.902 |                             0.825 |                    -1.245 |                              -3.268 |                          0.954 |
| GOOG |                   0.888 |                             0.824 |                    -0.636 |                              -1.085 |                          0.967 |
| MSFT |                   0.935 |                             0.865 |                    -7.146 |                              -3.585 |                          0.992 |
| TSLA |                   0.864 |                             0.842 |                     0.496 |                               0.857 |                          0.973 |

## Sentiment vs returns

|      |   corr_same_day |   p_same_day |   corr_next_day |   p_next_day |   Granger p: sentiment -> returns |   Granger p: returns -> sentiment |
|:-----|----------------:|-------------:|----------------:|-------------:|----------------------------------:|----------------------------------:|
| AAPL |          0.1645 |       0.0000 |          0.0303 |       0.2843 |                            0.0665 |                            0.0010 |
| AMZN |          0.1877 |       0.0000 |          0.0906 |       0.0013 |                            0.0016 |                            0.0311 |
| GOOG |          0.1101 |       0.0001 |          0.0541 |       0.0551 |                            0.0408 |                            0.0007 |
| MSFT |          0.1351 |       0.0000 |          0.0503 |       0.0745 |                            0.0002 |                            0.0577 |
| TSLA |          0.2086 |       0.0000 |          0.0496 |       0.0786 |                            0.0452 |                            0.1969 |

## Mean return by sentiment quintile (bp, averaged over stocks)

|   q |   ret |   ret_next |
|----:|------:|-----------:|
|   1 | -57.1 |      -13.9 |
|   2 |  -5.3 |        6.5 |
|   3 |  21.3 |       19.3 |
|   4 |  46.7 |       14.8 |
|   5 |  38.9 |       18.6 |

## Next-day forecasts, averaged over stocks

|                                      |   hit_rate |   excess_hit |   r2_oos |      ic |   p_beats_always_up |
|:-------------------------------------|-----------:|-------------:|---------:|--------:|--------------------:|
| ('price', 'Random forest')           |     0.5295 |      -0.0134 |  -0.0131 | -0.0048 |              0.9497 |
| ('price', 'Ridge')                   |     0.5335 |      -0.0093 |  -0.0036 |  0.0221 |              0.8739 |
| ('price', 'SVR (RBF)')               |     0.5221 |      -0.0207 |  -0.0450 |  0.0006 |              0.9944 |
| ('price+sentiment', 'Random forest') |     0.5286 |      -0.0143 |  -0.0064 |  0.0176 |              0.9592 |
| ('price+sentiment', 'Ridge')         |     0.5266 |      -0.0162 |  -0.0043 |  0.0335 |              0.9758 |
| ('price+sentiment', 'SVR (RBF)')     |     0.5196 |      -0.0232 |  -0.0400 |  0.0259 |              0.9977 |
| ('sentiment', 'Random forest')       |     0.5228 |      -0.0200 |  -0.0057 |  0.0221 |              0.9926 |
| ('sentiment', 'Ridge')               |     0.5195 |      -0.0233 |  -0.0035 |  0.0290 |              0.9977 |
| ('sentiment', 'SVR (RBF)')           |     0.5269 |      -0.0159 |  -0.0308 |  0.0349 |              0.9739 |

## Next-day forecasts, per stock

|                                                 |   hit_rate |   r2_oos |      ic |        n |
|:------------------------------------------------|-----------:|---------:|--------:|---------:|
| ('AAPL', 'none', 'Historical mean (always up)') |     0.5466 |   0.0000 | -0.0244 | 730.0000 |
| ('AAPL', 'price', 'Ridge')                      |     0.5068 |  -0.0103 |  0.0054 | 730.0000 |
| ('AAPL', 'price', 'SVR (RBF)')                  |     0.5178 |  -0.0728 | -0.0190 | 730.0000 |
| ('AAPL', 'price', 'Random forest')              |     0.5205 |  -0.0215 | -0.0251 | 730.0000 |
| ('AAPL', 'sentiment', 'Ridge')                  |     0.4849 |  -0.0114 | -0.0344 | 730.0000 |
| ('AAPL', 'sentiment', 'SVR (RBF)')              |     0.5219 |  -0.0230 |  0.0280 | 730.0000 |
| ('AAPL', 'sentiment', 'Random forest')          |     0.5192 |  -0.0131 |  0.0008 | 730.0000 |
| ('AAPL', 'price+sentiment', 'Ridge')            |     0.4849 |  -0.0225 | -0.0448 | 730.0000 |
| ('AAPL', 'price+sentiment', 'SVR (RBF)')        |     0.5178 |  -0.0443 |  0.0260 | 730.0000 |
| ('AAPL', 'price+sentiment', 'Random forest')    |     0.5027 |  -0.0256 | -0.0174 | 730.0000 |
| ('AMZN', 'none', 'Historical mean (always up)') |     0.5580 |   0.0000 |  0.0333 | 733.0000 |
| ('AMZN', 'price', 'Ridge')                      |     0.5539 |  -0.0057 |  0.0140 | 733.0000 |
| ('AMZN', 'price', 'SVR (RBF)')                  |     0.5280 |   0.0008 |  0.0588 | 733.0000 |
| ('AMZN', 'price', 'Random forest')              |     0.5484 |  -0.0207 | -0.0294 | 733.0000 |
| ('AMZN', 'sentiment', 'Ridge')                  |     0.5375 |   0.0010 |  0.0549 | 733.0000 |
| ('AMZN', 'sentiment', 'SVR (RBF)')              |     0.5184 |  -0.0577 | -0.0014 | 733.0000 |
| ('AMZN', 'sentiment', 'Random forest')          |     0.5443 |  -0.0040 |  0.0333 | 733.0000 |
| ('AMZN', 'price+sentiment', 'Ridge')            |     0.5607 |  -0.0027 |  0.0663 | 733.0000 |
| ('AMZN', 'price+sentiment', 'SVR (RBF)')        |     0.5075 |  -0.0550 | -0.0046 | 733.0000 |
| ('AMZN', 'price+sentiment', 'Random forest')    |     0.5389 |  -0.0108 |  0.0289 | 733.0000 |
| ('GOOG', 'none', 'Historical mean (always up)') |     0.5368 |   0.0000 | -0.0096 | 734.0000 |
| ('GOOG', 'price', 'Ridge')                      |     0.5572 |  -0.0092 |  0.0100 | 734.0000 |
| ('GOOG', 'price', 'SVR (RBF)')                  |     0.4782 |  -0.0759 | -0.0971 | 734.0000 |
| ('GOOG', 'price', 'Random forest')              |     0.5177 |  -0.0245 | -0.0347 | 734.0000 |
| ('GOOG', 'sentiment', 'Ridge')                  |     0.5504 |  -0.0026 |  0.0628 | 734.0000 |
| ('GOOG', 'sentiment', 'SVR (RBF)')              |     0.5463 |  -0.0371 |  0.0573 | 734.0000 |
| ('GOOG', 'sentiment', 'Random forest')          |     0.5341 |   0.0018 |  0.0517 | 734.0000 |
| ('GOOG', 'price+sentiment', 'Ridge')            |     0.5531 |  -0.0098 |  0.0427 | 734.0000 |
| ('GOOG', 'price+sentiment', 'SVR (RBF)')        |     0.5422 |  -0.0501 |  0.0137 | 734.0000 |
| ('GOOG', 'price+sentiment', 'Random forest')    |     0.5368 |  -0.0116 | -0.0176 | 734.0000 |
| ('MSFT', 'none', 'Historical mean (always up)') |     0.5667 |   0.0000 | -0.0299 | 727.0000 |
| ('MSFT', 'price', 'Ridge')                      |     0.5681 |   0.0180 |  0.1113 | 727.0000 |
| ('MSFT', 'price', 'SVR (RBF)')                  |     0.5695 |  -0.0447 |  0.0588 | 727.0000 |
| ('MSFT', 'price', 'Random forest')              |     0.5681 |   0.0150 |  0.0933 | 727.0000 |
| ('MSFT', 'sentiment', 'Ridge')                  |     0.5199 |  -0.0029 |  0.0271 | 727.0000 |
| ('MSFT', 'sentiment', 'SVR (RBF)')              |     0.5254 |  -0.0242 |  0.0486 | 727.0000 |
| ('MSFT', 'sentiment', 'Random forest')          |     0.5103 |  -0.0083 | -0.0027 | 727.0000 |
| ('MSFT', 'price+sentiment', 'Ridge')            |     0.5296 |   0.0248 |  0.1078 | 727.0000 |
| ('MSFT', 'price+sentiment', 'SVR (RBF)')        |     0.5502 |  -0.0227 |  0.1085 | 727.0000 |
| ('MSFT', 'price+sentiment', 'Random forest')    |     0.5433 |   0.0216 |  0.0781 | 727.0000 |
| ('TSLA', 'none', 'Historical mean (always up)') |     0.5061 |   0.0000 | -0.0461 | 733.0000 |
| ('TSLA', 'price', 'Ridge')                      |     0.4816 |  -0.0109 | -0.0302 | 733.0000 |
| ('TSLA', 'price', 'SVR (RBF)')                  |     0.5171 |  -0.0322 |  0.0013 | 733.0000 |
| ('TSLA', 'price', 'Random forest')              |     0.4925 |  -0.0139 | -0.0283 | 733.0000 |
| ('TSLA', 'sentiment', 'Ridge')                  |     0.5048 |  -0.0018 |  0.0344 | 733.0000 |
| ('TSLA', 'sentiment', 'SVR (RBF)')              |     0.5225 |  -0.0118 |  0.0420 | 733.0000 |
| ('TSLA', 'sentiment', 'Random forest')          |     0.5061 |  -0.0049 |  0.0275 | 733.0000 |
| ('TSLA', 'price+sentiment', 'Ridge')            |     0.5048 |  -0.0112 | -0.0045 | 733.0000 |
| ('TSLA', 'price+sentiment', 'SVR (RBF)')        |     0.4802 |  -0.0280 | -0.0143 | 733.0000 |
| ('TSLA', 'price+sentiment', 'Random forest')    |     0.5211 |  -0.0057 |  0.0159 | 733.0000 |
