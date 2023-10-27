
One of the difficulties in predicting precipitation pattern using deep learning models lies in the scarcity of precipitation; more than 90% of the data is area without precipitation.
The occurrence of heavy precipitations that could lead to societal damage or disruption is much more scarce.

We propose to utilize a deep learning-based “sampler”.
Given an input data sample, the sampler produces the probability of the data sample to be selected.
The network structure of the sampler is a convolutional network typically used for image classification tasks.
In the case of precipitation nowcasting, non-raining samples become worthless quickly after the initial period of training, since this type of data does not enhance understanding of the phenomena of interest.
On the contrary, samples of rarely occurring rainfall patterns, such as torrential precipitation, are weighted heavily.

The procedure is also called importance sampling.
Peng et al. (2019) is similar approach.

--
Reference:
Peng et al., Trainable Undersampling for Class-Imbalance Learning, Proceedings AAAI, 2019.
https://ojs.aaai.org/index.php/AAAI/article/view/4396


