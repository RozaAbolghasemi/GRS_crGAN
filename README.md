
# Adversarial Preference Learning with Pairwise Comparisons for Group Recommendation System
This is a Keras implementation of a group recommendation system, which is designed for comparison with the model introduced in the paper:
>R. Abolghasemi, E. Herrera Viedma, P. Engelstad, Y.Djenouri, A. Yazidi. A  Graph Neural Approach for Group Recommendation System Based on Pairwise Preferences. Information Fusion 2024

The model for the "Recommendation System" is described in the following paper, and the codes are available [Here](https://github.com/wang22ti/Adversarial-Preference-Learning-with-Pairwise-Comparisons)

>Z. Wang, Q. Xu, K. Ma, Y. Jiang, X. Cao and Q. Huang. Adversarial Preference Learning with Pairwise Comparisons. MM2019.
In this repository, we have modified the code and added additional functions to tailor it for "Group Recommendation Systems".

## Dependencies
The codebase is implemented in Python 3.10.9. package versions used for development are just below.
```
tensorflow             1.14.0
keras             2.3.0
numpy             1.23.4
pandas            1.5.3
scipy             1.10.0
```

 Install them by running.
```
pip install tensorflow==1.14.0
pip install keras==2.3.0
pip install numpy
pip install pandas
pip install matplotlib.pyplot
```
### Dataset
* Pairwise preference data: The dataset for the MFP method was acquired from an online experiment performed by [Blèdaitè et al.](https://dl.acm.org/doi/pdf/10.1145/2700171.2791049?casa_token=hjYzq9yecUsAAAAA:oR_T8e6uKVasBZ77VpqAGnzFi0jRk__jeiz9DkGq3ZTQa3TSIjiii_zfJBSidmQ5LM4PDhHqMw_i) to collect users’
pairwise preferences. The authors developed an online interface that allows users to compare different movie pairs and enter
their pairwise scores. In this experiment, a total of 2,262 pairwise scores related to 100 movies from the MovieLens dataset
were collected based on feedback from 46 users. In addition, 73,078 movie ratings from 1,128 users in the [MovieLens 100 K](https://grouplens.org/datasets/movielens/100k/)
dataset were used. These movie ratings were converted into pairwise scores. 

----------------------------------------------------------------------

**License**

[MIT License](https://github.com/RozaAbolghasemi/GRS_Personality_PairwiseComparison/blob/main/LICENSE)


----------------------------------------------------------------------

**Reference**

If you use this code/paper, please cite it as below.
```
>R. Abolghasemi, E. Herrera Viedma, P. Engelstad, Y.Djenouri, A. Yazidi. A  Graph Neural Approach for Group Recommendation System Based on Pairwise Preferences. Information Fusion 2024
```
