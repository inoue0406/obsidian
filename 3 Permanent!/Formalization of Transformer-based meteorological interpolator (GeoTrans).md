Let $R(x,y,t)$ be a meteorological field  with two-dimensional space and one-dimensional time dimensions. Unlike ordinary meteorological analysis, the field is usually taken to be spacious, so that multiple temporal and spatial scales are included. 

From the meteorological field $R(x,y,t)$, the values $R_{in}$ corresponding to $M$ discrete locations $x_{in},y_{in}$ and times $t_{in}$ are taken, and $R_{out}$ corresponding to $N$ discrete locations $x_{out},y_{out}$ and times $t_{out}$ are taken.
These values could be interpreted as *randomly* sampled observations of meteorological field. 
$R_{in}$ is used as the input, and $R_{out}$ is used as the output.

Unlike specialized network structures such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs), transformers by themselves cannot distinguish the ordering of input data. To overcome this problem, a positional encoding [[Ashish_2017_Attention_is_all_you_need]] is used. 
A positional encoding $PE(x,y,t)$ is a mapping from spatial and temporal locations to certain value varying in various temporal and spatial scales.

Using transformer encoder-decoder network structure the prediction is done in two steps. Firstly, an intermediate representation $Z$ is obtained from the input values $R_{in}$ and the positional encodings of input locations by the encoder $F_E$. Secondly, the output meteorological values $R_{out}$ is predicted from the intermediate representation and the positional encodings of output locations by the decoder $F_D$.
These steps are formulated as follows.

$$Z = F_E(R_{in},PE(x_{in},y_{in},t_{in}))$$
$$\tilde{R}_{out} = F_D(Z,PE(x_{out},y_{out},t_{out}))$$

Both the encoder and decoder are parameterized neural network using attention layers.
The model is trained by evaluating the difference between the predicted and the true values with a loss function,

$$L = L(\tilde{R}_{out},R_{out}),$$

then applying backpropagation to update the trainable parameters.


Reference:
[[Ashish_2017_Attention_is_all_you_need]]
