
# Adversarial Preference Learning with Pairwise Comparisons for Group Recommendation System
This is a Keras implementation of a group recommendation system using GAN, which is designed for comparison with the model introduced in the paper:
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
* Food dataset: The paper utilizes a food dataset from an online experiment, Consens@OsloMet, conducted at Oslo Metropolitan University (Norway), focusing on group decision making regarding food preferences. Participants were organized into groups of five and tasked with updating or maintaining their food choices based on the group's average opinion. The experiment, registered and approved by the Norwegian Centre for Research Data, involved an online interface where experts provided pairwise scores for different food pairs. The front-end interface displayed a probability score indicating their preferences. The collected data, representing preferences in matrices, enabled the study of consensus-building within groups. The paper's methodology and experimental design are detailed in the source, aiming to predict missing pairwise preferences in group decision making.

----------------------------------------------------------------------

**License**

[MIT License](https://github.com/RozaAbolghasemi/GRS_Personality_PairwiseComparison/blob/main/LICENSE)


----------------------------------------------------------------------

**Reference**

If you use this code/paper, please cite it as below.
```
>R. Abolghasemi, E. Herrera Viedma, P. Engelstad, Y.Djenouri, A. Yazidi. A  Graph Neural Approach for Group Recommendation System Based on Pairwise Preferences. Information Fusion 2024
```
