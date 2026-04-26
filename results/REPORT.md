# REPORT

## 1. Environment info
- Timestamp (UTC): 2026-04-26T19:39:17.823420+00:00
- Python: 3.11.14
- numpy: 1.26.4
- pandas: 2.3.3
- scipy: 1.16.3
- scikit-learn: 1.5.2

## 2. Dataset summary table
| name             |   n_samples |   n_features |   n_classes |   imbalance_ratio | source   | identifier                                      |
|:-----------------|------------:|-------------:|------------:|------------------:|:---------|:------------------------------------------------|
| iris             |         150 |            4 |           3 |           1       | sklearn  | load_iris                                       |
| wine             |         178 |           13 |           3 |           1.47917 | sklearn  | load_wine                                       |
| breast_cancer    |         569 |           30 |           2 |           1.68396 | sklearn  | load_breast_cancer                              |
| digits           |        1797 |           64 |          10 |           1.05172 | sklearn  | load_digits                                     |
| glass            |         214 |            9 |           6 |           8.44444 | openml   | name=glass,version=1                            |
| vehicle          |         846 |           18 |           4 |           1.09548 | openml   | name=vehicle,version=1                          |
| ionosphere       |         351 |           34 |           2 |           1.78571 | openml   | name=ionosphere,version=1                       |
| sonar            |         208 |           60 |           2 |           1.14433 | openml   | name=sonar,version=1                            |
| ecoli            |         336 |            7 |           8 |          71.5     | openml   | name=ecoli,version=1                            |
| yeast            |        1484 |            8 |          10 |          92.6     | openml   | name=yeast,version=1                            |
| segment          |        2310 |           19 |           7 |           1       | openml   | name=segment,version=1                          |
| waveform         |        5000 |           40 |           3 |           1.02359 | openml   | name=waveform-5000,version=1                    |
| optdigits        |        5620 |           64 |          10 |           1.03249 | openml   | name=optdigits,version=1                        |
| satellite        |        6430 |           36 |           6 |           2.4496  | openml   | name=satimage,version=1                         |
| pendigits        |       10992 |           16 |          10 |           1.08436 | openml   | name=pendigits,version=1                        |
| vowel            |         990 |           12 |          11 |           1       | openml   | name=vowel,version=2                            |
| balance_scale    |         625 |            4 |           3 |           5.87755 | openml   | name=balance-scale,version=1                    |
| page_blocks      |        5473 |           10 |           5 |         175.464   | openml   | name=page-blocks,version=1                      |
| spambase         |        4601 |           57 |           2 |           1.53778 | openml   | name=spambase,version=1                         |
| banknote         |        1372 |            4 |           2 |           1.24918 | openml   | name=banknote-authentication,version=1          |
| robot_navigation |        5456 |           24 |           4 |           6.72256 | openml   | name=wall-robot-navigation,version=1            |
| letter           |       20000 |           16 |          26 |           1.10763 | openml   | name=letter,version=1                           |
| transfusion      |         748 |            4 |           2 |           3.20225 | openml   | name=blood-transfusion-service-center,version=1 |
| parkinsons       |         195 |           22 |           2 |           3.0625  | openml   | name=parkinsons,version=1                       |

## 3. Headline accuracy table (best per row in bold)
| dataset          |   BernoulliNB |   ComplementNB | DW-NB(k=15,CV-lambda)   | DW-NB(k=15,lambda=0.5)   | DW-NB(k=30,lambda=0.5)   | DW-NB(k=5,lambda=0.5)   | DW-NB(w1-only)   | DW-NB(w2-only)   | DW-NB(w3-only)   | GaussianNB   |   MultinomialNB | NB+kNN-Ensemble   |
|:-----------------|--------------:|---------------:|:------------------------|:-------------------------|:-------------------------|:------------------------|:-----------------|:-----------------|:-----------------|:-------------|----------------:|:------------------|
| balance_scale    |        0.7601 |         0.8768 | **0.9040**              | 0.9008                   | 0.9024                   | 0.9024                  | 0.9024           | 0.9008           | **0.9040**       | **0.9040**   |          0.8768 | 0.9008            |
| banknote         |        0.8447 |         0.7362 | **0.9985**              | 0.9964                   | 0.9905                   | 0.9978                  | 0.9964           | 0.9934           | 0.9964           | 0.8382       |          0.6363 | 0.9934            |
| breast_cancer    |        0.9315 |         0.8506 | **0.9666**              | 0.9351                   | 0.9333                   | 0.9456                  | 0.9351           | 0.9351           | 0.9421           | 0.9315       |          0.8401 | 0.9351            |
| digits           |        0.8853 |         0.8197 | **0.9761**              | 0.8792                   | 0.8581                   | 0.9104                  | 0.8770           | 0.8765           | 0.8826           | 0.7891       |          0.9004 | 0.8692            |
| ecoli            |        0.7772 |         0.6789 | 0.8154                  | **0.8187**               | **0.8187**               | **0.8187**              | **0.8187**       | **0.8187**       | **0.8187**       | 0.7439       |          0.4256 | **0.8187**        |
| glass            |        0.6595 |         0.4959 | **0.7076**              | 0.5844                   | 0.5844                   | 0.6126                  | 0.5844           | 0.5799           | 0.6078           | 0.4578       |          0.4636 | 0.5989            |
| ionosphere       |        0.7606 |         0.7551 | 0.8832                  | 0.8946                   | 0.8860                   | 0.8946                  | 0.8946           | 0.8917           | **0.8975**       | 0.8832       |          0.6552 | 0.8917            |
| iris             |        0.7533 |         0.6667 | 0.9400                  | **0.9533**               | **0.9533**               | **0.9533**              | **0.9533**       | 0.9467           | **0.9533**       | **0.9533**   |          0.7933 | 0.9467            |
| letter           |        0.4215 |         0.4064 | **0.9484**              | 0.8524                   | 0.8162                   | 0.9132                  | 0.8482           | 0.8420           | 0.8713           | 0.6436       |          0.524  | 0.8505            |
| optdigits        |        0.8952 |         0.821  | **0.9774**              | 0.8587                   | 0.8315                   | 0.8801                  | 0.8557           | 0.8555           | 0.8582           | 0.7226       |          0.9071 | 0.8952            |
| page_blocks      |        0.8904 |         0.9097 | **0.9647**              | 0.9476                   | 0.9445                   | 0.9538                  | 0.9503           | 0.9476           | 0.9481           | 0.9004       |          0.8986 | 0.9428            |
| parkinsons       |        0.7187 |         0.7076 | **0.8924**              | 0.7134                   | 0.7032                   | 0.8416                  | 0.7134           | 0.7084           | 0.7184           | 0.6982       |          0.8511 | 0.7084            |
| pendigits        |        0.8105 |         0.6995 | **0.9910**              | 0.9684                   | 0.9502                   | 0.9812                  | 0.9677           | 0.9677           | 0.9696           | 0.8564       |          0.8015 | 0.9694            |
| robot_navigation |        0.6067 |         0.596  | **0.8660**              | 0.7403                   | 0.6774                   | 0.8297                  | 0.7505           | 0.7231           | 0.7564           | 0.5278       |          0.5902 | 0.6979            |
| satellite        |        0.6935 |         0.5414 | **0.9070**              | 0.8417                   | 0.8297                   | 0.8607                  | 0.8387           | 0.8384           | 0.8412           | 0.7958       |          0.5793 | 0.8498            |
| segment          |        0.7234 |         0.4957 | **0.9450**              | 0.8835                   | 0.8442                   | 0.9333                  | 0.8970           | 0.8831           | 0.8887           | 0.7961       |          0.7693 | 0.8727            |
| sonar            |        0.7545 |         0.7452 | **0.7931**              | 0.7021                   | 0.6926                   | 0.7783                  | 0.7069           | 0.7069           | 0.7212           | 0.6826       |          0.7548 | 0.7069            |
| spambase         |        0.9026 |         0.8368 | **0.9231**              | 0.8268                   | 0.8224                   | 0.8376                  | 0.8274           | 0.8263           | 0.8274           | 0.8166       |          0.8898 | 0.8520            |
| transfusion      |        0.7473 |         0.6951 | **0.7834**              | 0.7647                   | 0.7606                   | 0.7726                  | 0.7553           | 0.7620           | 0.7674           | 0.7446       |          0.7621 | 0.7620            |
| vehicle          |        0.4575 |         0.4645 | **0.7022**              | 0.6242                   | 0.5828                   | 0.6690                  | 0.6183           | 0.6183           | 0.6419           | 0.4504       |          0.5367 | 0.6112            |
| vowel            |        0.3768 |         0.3242 | 0.9182                  | 0.8465                   | 0.7980                   | **0.9545**              | 0.8606           | 0.7778           | 0.8970           | 0.6747       |          0.4616 | 0.7545            |
| waveform         |        0.7808 |         0.7288 | **0.8252**              | 0.8088                   | 0.8088                   | 0.8082                  | 0.8100           | 0.8098           | 0.8096           | 0.7980       |          0.8164 | 0.8098            |
| wine             |        0.9271 |         0.865  | 0.9833                  | 0.9833                   | 0.9778                   | **0.9889**              | 0.9833           | 0.9833           | 0.9778           | 0.9778       |          0.9441 | 0.9833            |
| yeast            |        0.4724 |         0.5168 | **0.5627**              | 0.4919                   | 0.4919                   | 0.4919                  | 0.5182           | 0.4919           | 0.5128           | 0.1348       |          0.3329 | 0.3012            |

## 4. lambda analysis (DWGaussianNB_CV)
| dataset          |   mean_lambda |   std_lambda |
|:-----------------|--------------:|-------------:|
| balance_scale    |          0.23 |    0.11595   |
| banknote         |          0.88 |    0.0632456 |
| breast_cancer    |          1    |    0         |
| digits           |          1    |    0         |
| ecoli            |          0.32 |    0.193218  |
| glass            |          0.98 |    0.0421637 |
| ionosphere       |          0.3  |    0.270801  |
| iris             |          0.45 |    0.417     |
| letter           |          1    |    0         |
| optdigits        |          1    |    0         |
| page_blocks      |          1    |    0         |
| parkinsons       |          1    |    0         |
| pendigits        |          1    |    0         |
| robot_navigation |          1    |    0         |
| satellite        |          1    |    0         |
| segment          |          0.99 |    0.0316228 |
| sonar            |          0.92 |    0.0421637 |
| spambase         |          1    |    0         |
| transfusion      |          0.83 |    0.11595   |
| vehicle          |          1    |    0         |
| vowel            |          0.86 |    0.0699206 |
| waveform         |          0.89 |    0.0316228 |
| wine             |          0.65 |    0.356682  |
| yeast            |          1    |    0         |

## 5. NB-kNN agreement rate table
| dataset          |   nb_knn_agreement_rate |   accuracy_gain |   spearman_rho |   spearman_pvalue |
|:-----------------|------------------------:|----------------:|---------------:|------------------:|
| balance_scale    |                0.953661 |     -0.0032002  |      -0.755652 |       1.95728e-05 |
| banknote         |                0.838242 |      0.158109   |      -0.755652 |       1.95728e-05 |
| breast_cancer    |                0.954355 |      0.00350877 |      -0.755652 |       1.95728e-05 |
| digits           |                0.797449 |      0.0901397  |      -0.755652 |       1.95728e-05 |
| ecoli            |                0.774064 |      0.0747772  |      -0.755652 |       1.95728e-05 |
| glass            |                0.507576 |      0.126623   |      -0.755652 |       1.95728e-05 |
| ionosphere       |                0.843095 |      0.0114286  |      -0.755652 |       1.95728e-05 |
| iris             |                0.966667 |      0          |      -0.755652 |       1.95728e-05 |
| letter           |                0.65     |      0.20875    |      -0.755652 |       1.95728e-05 |
| optdigits        |                0.730961 |      0.136121   |      -0.755652 |       1.95728e-05 |
| page_blocks      |                0.903889 |      0.0471423  |      -0.755652 |       1.95728e-05 |
| parkinsons       |                0.682105 |      0.0152632  |      -0.755652 |       1.95728e-05 |
| pendigits        |                0.857716 |      0.112081   |      -0.755652 |       1.95728e-05 |
| robot_navigation |                0.543431 |      0.212433   |      -0.755652 |       1.95728e-05 |
| satellite        |                0.835614 |      0.0458787  |      -0.755652 |       1.95728e-05 |
| segment          |                0.794372 |      0.0874459  |      -0.755652 |       1.95728e-05 |
| sonar            |                0.701905 |      0.0195238  |      -0.755652 |       1.95728e-05 |
| spambase         |                0.803951 |      0.010216   |      -0.755652 |       1.95728e-05 |
| transfusion      |                0.852865 |      0.0201081  |      -0.755652 |       1.95728e-05 |
| vehicle          |                0.511905 |      0.173852   |      -0.755652 |       1.95728e-05 |
| vowel            |                0.651515 |      0.171717   |      -0.755652 |       1.95728e-05 |
| waveform         |                0.8206   |      0.0108     |      -0.755652 |       1.95728e-05 |
| wine             |                0.944118 |      0.00555556 |      -0.755652 |       1.95728e-05 |
| yeast            |                0.146263 |      0.357065   |      -0.755652 |       1.95728e-05 |

## 6. Weight component ablation summary
| dataset          |   acc_w1 |   acc_w2 |   acc_w3 |   acc_all_three | best_single_component   |   best_single_acc | all_three_ge_best_single   |   all_minus_best_single |
|:-----------------|---------:|---------:|---------:|----------------:|:------------------------|------------------:|:---------------------------|------------------------:|
| balance_scale    | 0.902355 | 0.900768 | 0.903968 |        0.900768 | w3                      |          0.903968 | False                      |            -0.0032002   |
| banknote         | 0.99635  | 0.993436 | 0.99635  |        0.99635  | w1                      |          0.99635  | True                       |             0           |
| breast_cancer    | 0.935056 | 0.935056 | 0.942074 |        0.935056 | w3                      |          0.942074 | False                      |            -0.00701754  |
| digits           | 0.877008 | 0.876453 | 0.882576 |        0.879236 | w3                      |          0.882576 | False                      |            -0.00333954  |
| ecoli            | 0.818717 | 0.818717 | 0.818717 |        0.818717 | w1                      |          0.818717 | True                       |             0           |
| glass            | 0.584416 | 0.57987  | 0.607792 |        0.584416 | w3                      |          0.607792 | False                      |            -0.0233766   |
| ionosphere       | 0.894603 | 0.891746 | 0.89746  |        0.894603 | w3                      |          0.89746  | False                      |            -0.00285714  |
| iris             | 0.953333 | 0.946667 | 0.953333 |        0.953333 | w1                      |          0.953333 | True                       |             0           |
| letter           | 0.84825  | 0.842    | 0.87135  |        0.8524   | w3                      |          0.87135  | False                      |            -0.01895     |
| optdigits        | 0.855694 | 0.855516 | 0.858185 |        0.858719 | w3                      |          0.858185 | True                       |             0.000533808 |
| page_blocks      | 0.950299 | 0.947557 | 0.948106 |        0.947558 | w1                      |          0.950299 | False                      |            -0.00274056  |
| parkinsons       | 0.713421 | 0.708421 | 0.718421 |        0.713421 | w3                      |          0.718421 | False                      |            -0.005       |
| pendigits        | 0.967705 | 0.967705 | 0.969615 |        0.968433 | w3                      |          0.969615 | False                      |            -0.00118265  |
| robot_navigation | 0.750544 | 0.723053 | 0.756411 |        0.740283 | w3                      |          0.756411 | False                      |            -0.016128    |
| satellite        | 0.838725 | 0.838414 | 0.841213 |        0.84168  | w3                      |          0.841213 | True                       |             0.000466563 |
| segment          | 0.89697  | 0.883117 | 0.888745 |        0.88355  | w1                      |          0.89697  | False                      |            -0.0134199   |
| sonar            | 0.706905 | 0.706905 | 0.72119  |        0.702143 | w3                      |          0.72119  | False                      |            -0.0190476   |
| spambase         | 0.827428 | 0.826342 | 0.827429 |        0.826776 | w3                      |          0.827429 | False                      |            -0.000652174 |
| transfusion      | 0.755261 | 0.762    | 0.767369 |        0.764667 | w3                      |          0.767369 | False                      |            -0.0027027   |
| vehicle          | 0.618333 | 0.618319 | 0.641905 |        0.624202 | w3                      |          0.641905 | False                      |            -0.0177031   |
| vowel            | 0.860606 | 0.777778 | 0.89697  |        0.846465 | w3                      |          0.89697  | False                      |            -0.0505051   |
| waveform         | 0.81     | 0.8098   | 0.8096   |        0.8088   | w1                      |          0.81     | False                      |            -0.0012      |
| wine             | 0.983333 | 0.983333 | 0.977778 |        0.983333 | w1                      |          0.983333 | True                       |             0           |
| yeast            | 0.518153 | 0.491869 | 0.512765 |        0.491878 | w1                      |          0.518153 | False                      |            -0.0262743   |

Combining all three components is >= best individual on 6/24 datasets.

## 7. Win/tie/loss counts
| baseline               |   wins |   ties |   losses |
|:-----------------------|-------:|-------:|---------:|
| BernoulliNB            |     18 |      0 |        6 |
| ComplementNB           |     21 |      0 |        3 |
| DW-NB(k=15,CV-lambda)  |      3 |      1 |       20 |
| DW-NB(k=30,lambda=0.5) |     18 |      5 |        1 |
| DW-NB(k=5,lambda=0.5)  |      1 |      4 |       19 |
| DW-NB(w1-only)         |      7 |      8 |        9 |
| DW-NB(w2-only)         |     18 |      5 |        2 |
| DW-NB(w3-only)         |      3 |      3 |       18 |
| GaussianNB             |     22 |      1 |        1 |
| MultinomialNB          |     18 |      0 |        6 |
| NB+kNN-Ensemble        |     13 |      4 |        7 |

## 8. Geometric interpolation vs arithmetic averaging
| comparison                                |   wins |   ties |   losses |
|:------------------------------------------|-------:|-------:|---------:|
| DW-NB(k=15,lambda=0.5) vs NB+kNN-Ensemble |     13 |      4 |        7 |

## 9. Friedman p-values and average ranks
| metric                     |   friedman_statistic |     p_value |   n_datasets |
|:---------------------------|---------------------:|------------:|-------------:|
| accuracy                   |             141.048  | 9.96776e-25 |           24 |
| auc_roc                    |              95.6887 | 1.26985e-15 |           16 |
| balanced_accuracy          |             141.819  | 6.94658e-25 |           24 |
| brier_score                |             138.601  | 3.13626e-24 |           24 |
| ece                        |              83.3068 | 3.3732e-13  |           24 |
| geometric_mean             |             146.794  | 6.7269e-26  |           24 |
| log_loss                   |              81.5471 | 7.4048e-13  |           24 |
| macro_f1                   |             140.971  | 1.03343e-24 |           24 |
| mcc                        |             137.669  | 4.84946e-24 |           24 |
| predict_time_per_sample_ms |             187.891  | 2.39964e-34 |           24 |
| weighted_f1                |             138.548  | 3.21401e-24 |           24 |

| metric      | classifier             |   avg_rank |
|:------------|:-----------------------|-----------:|
| accuracy    | DW-NB(k=15,CV-lambda)  |    2.14583 |
| accuracy    | DW-NB(k=5,lambda=0.5)  |    3.02083 |
| accuracy    | DW-NB(w3-only)         |    4.1875  |
| accuracy    | DW-NB(w1-only)         |    5.20833 |
| accuracy    | DW-NB(k=15,lambda=0.5) |    5.45833 |
| accuracy    | NB+kNN-Ensemble        |    6.25    |
| accuracy    | DW-NB(w2-only)         |    6.83333 |
| accuracy    | DW-NB(k=30,lambda=0.5) |    7.75    |
| accuracy    | MultinomialNB          |    8.41667 |
| accuracy    | BernoulliNB            |    8.60417 |
| accuracy    | GaussianNB             |    9.79167 |
| accuracy    | ComplementNB           |   10.3333  |
| auc_roc     | DW-NB(k=15,CV-lambda)  |    3.34375 |
| auc_roc     | NB+kNN-Ensemble        |    3.34375 |
| auc_roc     | DW-NB(k=5,lambda=0.5)  |    3.4375  |
| auc_roc     | DW-NB(w3-only)         |    4.71875 |
| auc_roc     | DW-NB(w1-only)         |    5.34375 |
| auc_roc     | DW-NB(k=15,lambda=0.5) |    5.4375  |
| auc_roc     | DW-NB(w2-only)         |    6.84375 |
| auc_roc     | DW-NB(k=30,lambda=0.5) |    7.0625  |
| auc_roc     | GaussianNB             |    8.78125 |
| auc_roc     | MultinomialNB          |    9.375   |
| auc_roc     | BernoulliNB            |    9.65625 |
| auc_roc     | ComplementNB           |   10.6562  |
| brier_score | DW-NB(k=15,CV-lambda)  |    2.33333 |
| brier_score | NB+kNN-Ensemble        |    3.875   |
| brier_score | DW-NB(k=5,lambda=0.5)  |    4.04167 |
| brier_score | DW-NB(w3-only)         |    4.5     |
| brier_score | DW-NB(k=15,lambda=0.5) |    5.5625  |
| brier_score | DW-NB(w1-only)         |    5.70833 |
| brier_score | DW-NB(w2-only)         |    6.75    |
| brier_score | BernoulliNB            |    7.75    |
| brier_score | DW-NB(k=30,lambda=0.5) |    8.0625  |
| brier_score | MultinomialNB          |    8.625   |
| brier_score | GaussianNB             |   10.0833  |
| brier_score | ComplementNB           |   10.7083  |
| ece         | DW-NB(k=15,CV-lambda)  |    3.125   |
| ece         | DW-NB(k=5,lambda=0.5)  |    4.25    |
| ece         | NB+kNN-Ensemble        |    4.83333 |
| ece         | DW-NB(w3-only)         |    5.08333 |
| ece         | DW-NB(w1-only)         |    6.08333 |
| ece         | DW-NB(k=15,lambda=0.5) |    6.47917 |
| ece         | DW-NB(w2-only)         |    6.625   |
| ece         | BernoulliNB            |    6.79167 |
| ece         | MultinomialNB          |    7.25    |
| ece         | DW-NB(k=30,lambda=0.5) |    8.10417 |
| ece         | GaussianNB             |    9.5     |
| ece         | ComplementNB           |    9.875   |
| log_loss    | NB+kNN-Ensemble        |    2.95833 |
| log_loss    | DW-NB(k=15,CV-lambda)  |    3.58333 |
| log_loss    | DW-NB(w3-only)         |    5.70833 |
| log_loss    | MultinomialNB          |    6       |
| log_loss    | DW-NB(w1-only)         |    6.125   |
| log_loss    | DW-NB(k=5,lambda=0.5)  |    6.20833 |
| log_loss    | DW-NB(k=15,lambda=0.5) |    6.4375  |
| log_loss    | DW-NB(w2-only)         |    7.16667 |
| log_loss    | BernoulliNB            |    7.25    |
| log_loss    | DW-NB(k=30,lambda=0.5) |    7.89583 |
| log_loss    | ComplementNB           |    8.125   |
| log_loss    | GaussianNB             |   10.5417  |
| macro_f1    | DW-NB(k=15,CV-lambda)  |    2.20833 |
| macro_f1    | DW-NB(k=5,lambda=0.5)  |    3.02083 |
| macro_f1    | DW-NB(w3-only)         |    4.02083 |
| macro_f1    | DW-NB(w1-only)         |    5.08333 |
| macro_f1    | DW-NB(k=15,lambda=0.5) |    5.5     |
| macro_f1    | NB+kNN-Ensemble        |    6.35417 |
| macro_f1    | DW-NB(w2-only)         |    6.91667 |
| macro_f1    | DW-NB(k=30,lambda=0.5) |    7.72917 |
| macro_f1    | BernoulliNB            |    8.29167 |
| macro_f1    | MultinomialNB          |    9       |
| macro_f1    | GaussianNB             |    9.5     |
| macro_f1    | ComplementNB           |   10.375   |

## 10. Notable observations
- DW-NB(CV-λ) mean fold runtime is 37.99x of fixed (fixed=0.5520s, cv=20.9687s).
- Spearman correlation (agreement vs gain): rho=-0.7557, p=1.957e-05.