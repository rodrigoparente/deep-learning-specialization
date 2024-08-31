# Sequence Models

 - Sequence models are a type of neural network designed to handle sequential data, where the order of the inputs and outputs is important.
 - These models are particularly useful for tasks involving time series, natural language, speech, or any data that unfolds over time. 
 - Common applications include:
    - **Speech Recognition**: Maps an audio clip $ X $ (sequence) to a text transcript $ Y $ (sequence of words).
    - **Music Generation**: Generates a sequence of music notes $ Y $ from an initial input $ X $ (which can be empty, an integer, or a few notes).
    - **Sentiment Classification**: Analyzes a phrase $ X $ (sequence of words) to predict a sentiment score $ Y $ (e.g., star rating).
    - **DNA Sequence Analysis**: Labels a DNA sequence $ X $ (sequence of A, C, G, T) to identify specific segments $ Y $ (e.g., protein regions).
    - **Machine Translation**: Translates a sentence $ X $ (sequence of words in one language) into another language $ Y $ (sequence in the target language).
    - **Video Activity Recognition**: Recognizes an activity $ Y $ based on a sequence of video frames $ X $.
    - **Named Entity Recognition (NER)**: Identifies entities $ Y $ (e.g., names of people) within a given sentence $ X $ (sequence of words).

# Notation

 - Consider the input sentence "Harry Potter and Hermione Granger invented a new spell".
 - The goal of the sequence model is to identify the names "Harry Potter" and "Hermione Granger".
 - In this scenario, the input sequence $X$, of lenght $T_x$ (i.e., the number of words), consists of a series of words.
 - Each word in $X$ is indexed by $X^{\langle t \rangle}$, with $t$ indicating its position in the sequence.
 - The output sequence $Y$, of lenght $T_y$ (typically equal to $T_x$), provides corresponding labels for each word in $X$.
 - Additionally:
    - $X^{(i)\langle t \rangle}$ refers to the $i$-th training example in the dataset.
    - $T_x^{(i)}$ is the length of the input sequence for the $i$-th training example.
    - $Y^{(i)\langle t \rangle}$ is the label for the $t$-th word in the $i$-th training example.
    - $T_y^{(i)}$ is the length of the output sequence for the $i$-th training example.
  
**Vocabulary and Word Representation**
 
 - Vocabulary or dictionary is a list of words used to represent the input sequence, where each word is assigned a unique index.
 - Each word in the sequence is represented as a one-hot vector with a single '1' at the position corresponding to the word's index in the vocabulary, and '0's elsewhere.
 - If a word is not in the vocabulary, it is represented by a special token "UNK" (Unknown Word), which has its own one-hot vector.

# Recurrent Neural Network Model
 
 - RNNs process sequences step-by-step, passing information from one time step to the next.
 - For example, when reading a sentence, the model processes the first word $X^{\langle 1 \rangle}$ and passes this information to the next time step where it processes $X^{\langle 2 \rangle}$, and so on.
 - The hidden state from the previous time step, $a^{\langle t-1 \rangle}$, is used in conjunction with the current input $X^{\langle t \rangle}$ to predict the output $Y^{\langle t \rangle}$.
 - To initiate the process, the initial hidden state $a^{\langle 0 \rangle}$ is usually set to a vector of zeros.

**Parameter Sharing in RNNs**

 - RNNs share the same parameters across all time steps, allowing them to generalize features learned at one position to others:
 - **Weight Matrices**:
    - $W_{ax}$ is used to connect the input $X^{\langle t \rangle}$ to the hidden layer $a^{\langle t \rangle}$.
    - $W_{aa}$ connects the hidden state from the previous time step $a^{\langle t-1 \rangle}$ to the current hidden state $a^{\langle t \rangle}$.
    - $W_{ya}$ connects the hidden state $a^{\langle t \rangle}$ to the output $Y^{\langle t \rangle}$.
 - **Bias Terms**:
    - Bias terms $b_a$ and $b_y$ are added during the computation of the hidden state and output, respectively.

**Computation in RNNs**

 - The hidden state at time $t$ is computed as:
    $$ a^{\langle t \rangle} = g(W_{aa}a^{\langle t-1 \rangle} + W_{ax}X^{\langle t \rangle} + b_a) $$
    where $g$ is the activation function, often chosen as $\tanh$ for RNNs.
  - The output at time $t$ is computed as:
    $$ \hat{Y}^{\langle t \rangle} = g(W_{ya}a^{\langle t \rangle} + b_y) $$
    where $g$ could be a sigmoid or softmax function depending on the output type.

**Simplification of Notation**

 - To simplify the notation, the weight matrices $W_{aa}$ and $W_{ax}$ can be combined into a single matrix $W_a$ such that:
    $$ W_a = \begin{bmatrix} W_{aa} & W_{ax} \end{bmatrix} $$
 - This allows the hidden state computation to be written as:
    $$ a^{\langle t \rangle} = g(W_a \begin{bmatrix} a^{\langle t-1 \rangle} \\ X^{\langle t \rangle} \end{bmatrix} + b_a) $$
 - The output computation remains:
    $$ \hat{Y}^{\langle t \rangle} = g(W_{ya}a^{\langle t \rangle} + b_y) $$

**Limitations of Basic RNNs**

 - Basic RNNs only use information from earlier time steps to predict the current output. 
 - This can be problematic in cases where future context is necessary to accurately predict an output (e.g., determining if "Teddy" refers to a person or a bear).

# Backpropagation Through Time

 - For a single timestep $ t $, the loss is computed using the cross-entropy loss:
    $$ L^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) = -y^{\langle t \rangle}\log\hat{y}^{\langle t \rangle} - (1 - y^{\langle t \rangle})\log(1 - \hat{y}^{\langle t \rangle}) $$
 - The loss for timestep $ t $ is calculated based on the prediction $ \hat{y}^{\langle t \rangle} $ and the true label $ y^{\langle t \rangle} $.
 - The total loss $ L $ for the sequence is the sum of the losses across all timesteps: 
    $$ L(\hat{y}, y) = \sum_{t=1}^{T_x} L^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) $$
 - Backpropagation is performed in the reverse direction of forward propagation, from right to left across the timesteps.
 - Gradients are computed for the loss with respect to the parameters $ W_a, b_a, W_y, $ and $ b_y $.
- These gradients are then used to update the parameters via gradient descent.
  - The backpropagation process in RNNs is called "Backpropagation Through Time" (BPTT) because it involves reversing the sequence of operations over time.
  - The name suggests moving backward through time, as backpropagation follows the time indices in reverse.

# Different Types of RNNs

 1. **Many-to-Many Architecture**:
    - **Input:** Sequence $ x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \dots, x^{\langle T_x \rangle} $.
    - **Output:** Sequence $ \hat{y}^{\langle 1 \rangle}, \hat{y}^{\langle 2 \rangle}, \dots, \hat{y}^{\langle T_y \rangle} $.
    - Used when $ T_x = T_y $, such as in tasks where the input and output sequences are of the same length.
  
 2. **Many-to-One Architecture**:
    - **Input:** A text sequence (e.g., "There is nothing to like in this movie").
    - **Output:** A single value (e.g., an integer from 1 to 5).
    - The RNN reads the entire input sequence and produces an output only at the final timestep, usually used for classification.

 3. **One-to-Many Architecture**:
    - **Input:** A single value $ x $ (e.g., a genre or the first note).
    - **Output:** A sequence of values $ \hat{y}^{\langle 1 \rangle}, \hat{y}^{\langle 2 \rangle}, \dots $ (e.g., notes in a musical piece).
    - The RNN generates an entire sequence from a single input, commonly used in sequence generation tasks.

 4. **One-to-One Architecture**:
    - **Input:** A single value $ x $.
    - **Output:** A single value $ y $.
    - This is equivalent to a standard feedforward neural network, often used for simpler tasks where no sequence processing is needed.

# Language Model and Sequence Generation

 - A language model estimates the probability of a given sentence.
 - It helps in tasks like speech recognition and machine translation by determining which sentences are more likely.

**Common Terminology**

 - Corpus: A large collection of text used for training.
 - Tokenization: Converts sentences into tokens or indices.
 - End-of-Sentence $\langle$ EOS $\rangle$ Token: Helps in detecting sentence boundaries (optional).
 - Vocabulary: A set of common words and tokens.
 - Handling Unknown Words: Use an $\langle$ UNK $\rangle$ token for words not in the vocabulary.

**How RNN Works?**

 - Given as input a sequence of words $ y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \ldots, y^{\langle t \rangle} $.
 - Its goal is to estimates the probability of the entire sequence.
 - **Initialization:**
    - At time step $ t = 0 $: Initialize activations $ a^{\langle 0 \rangle} $ and inputs $ x^{\langle 1 \rangle} $ (e.g., as zero vectors).
 - **Forward Propagation**
    - At each time step $ t $:
        - Compute activation $ a^{\langle t \rangle} $ based on previous activation $ a^{\langle t - 1 \rangle} $ and current input $ x^{\langle t \rangle} $.
        - Predict the probability distribution of the next word using Softmax.
        - Continue this process until the end of the sequence.
 - **Loss Function**
    - Measures the difference between the predicted probability and the actual word.
    - Sum of losses across all time steps.
 - **Backward Propagation**
    - Adjust model parameters using gradient descent to minimize the loss.

# Sampling Novel Sequences

 1. **Initialization**:
    - Start with an initial input $ x^{\langle 1 \rangle} = 0 $ and initial activation $ a^{\langle 0 \rangle} = 0 $.
    - At the first time step, use Softmax to compute the probability distribution over possible outputs.

 2. **First Word Sampling**:
    - Sample the first word $ y^{\langle 1 \rangle} $ from the distribution defined by the Softmax output.
    - Use functions like `np.random.choice` to select a word based on the Softmax probabilities.

 3. **Subsequent Words**:
    - For each subsequent time step $ t $:
        - Set $ x^{\langle t \rangle} = y^{\langle t - 1 \rangle} $ (the previously sampled word).
        - Compute the new activation $ a^{\langle t \rangle} $ and output probabilities using Softmax.
        - Sample the next word $ y^{\langle t \rangle} $ based on the updated Softmax distribution.

 4. **Ending the Sequence**:
    - Continue sampling until an end-of-sentence (EOS) token is generated, if it is part of the vocabulary.
    - Set a fixed length (e.g., 20 or 100 words) and stop after generating that number of words.

# Character-Level vs. Word-Level Models

**Character-Level Models**

 - Includes characters (a-z, A-Z, space, punctuation, digits).
 - Sequence is made up of individual characters instead of words.
 - Advantages:
    - No need for an unknown word token.
    - Can model sequences with unseen words (e.g., "mau").
 - Disadvantages:
    - Generates much longer sequences.
    - Less effective at capturing long-range dependencies.
    - More computationally expensive to train.

**Word-Level Models**

 - Includes words or tokens from the training text.
 - Sequence is made up of individual words.
 - Advantages:
    - Typically better at capturing long-range dependencies.
    - More efficient to train compared to character-level models.
 - Disadvantages:
    - Require a large vocabulary which increases computational requirements and memory usage.
    - Rare or domain-specific words may not appear frequently enough to be included.

# Vanishing Gradients with RNNs

 - The vanishing gradient problem in RNNs happens when gradients become very small as they are propagated back through many time steps during training.
 - As a result, the network has difficulty updating weights for earlier time steps effectively.
 - This leads to problems in capturing and retaining long-range dependencies in sequences, since the influence of earlier inputs diminishes over time.
 - Consequently, RNNs struggle to learn patterns that span long sequences because the gradients needed to adjust weights in earlier layers are too small to be effective.

# Gated Recurrent Unit (GRU)

 - GRUs introduce a new variable, $ c^{\langle t \rangle} $, called the memory cell, which helps retain information over long sequences.
 - At each time step, the GRU outputs an activation $ a^{\langle t \rangle} $, which is equal to the memory cell value $ c^{\langle t \rangle} $.
 - The memory cell $ c^{\langle t \rangle} $ is updated based on a candidate value $ \tilde{c}^{\langle t \rangle} $, which is computed as:
    $$ \tilde{c}^{\langle t \rangle} = \tanh(W_c [c^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b_c) $$
 - The update of $ c^{\langle t \rangle} $ is controlled by a gate, $ \Gamma_u $, known as the update gate:
    $$ c^{\langle t \rangle} = \Gamma_u \ast \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u) \ast c^{\langle t - 1 \rangle} $$
 - The update gate $ \Gamma_u $ is computed using a sigmoid function, ensuring its value lies between 0 and 1:
    $$ \Gamma_u = \sigma(W_u [c^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b_u) $$

**Full GRU Unit**

 - In the full GRU model, an additional gate $ \Gamma_r $ (relevance gate) is introduced:
    $$ \tilde{c}^{\langle t \rangle} = \tanh(W_c [\Gamma_r \ast c^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b) $$
  - $ \Gamma_r $ determines how much of the previous memory cell $ c^{\langle t - 1 \rangle} $ is relevant for computing the new candidate value $ \tilde{c}^{\langle t \rangle} $.
  - The update gate $ \Gamma_r $ is computed by:
    $$ \Gamma_r = \sigma(W_r [c^{\langle t - 1 \rangle}, x^{\langle t \rangle}] + b_r) $$

# Long Short Term Memory (LSTM)

 - The LSTM (Long Short-Term Memory) is a more powerful and general version of the GRU.
 - Developed by Sepp Hochreiter and Jürgen Schmidhuber, the LSTM paper had a significant impact on sequence modeling.
 - LSTM’s equations are more complex compared to GRU, with additional gates and operations.

**Updated Operations**

 - The update gate $ \Gamma_u $ decides how much of the new information $ \tilde{c}^{\langle t-1 \rangle} $ should be added.
    $$ \Gamma_u = \sigma(W_u[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u) $$
 - The forget gate $ \Gamma_f $, controls the extent to which the previous memory cell $ c^{\langle t-1 \rangle} $ is retained:
    $$ \Gamma_f = \sigma(W_f[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f) $$
 - Output gate $\Gamma_o$ determines the output of the LSTM unit:
    $$ \Gamma_o = \sigma(W_o[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o) $$
 - The candidate memory $ \tilde{c}^{\langle t \rangle} $ is computed using:
    $$ \tilde{c}^{\langle t \rangle} = \tanh(W_c[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c) $$
 - The memory cell update is computed as:
    $$ c^{\langle t \rangle} = \Gamma_u \ast \tilde{c}^{\langle t \rangle} + \Gamma_f \ast c^{\langle t-1 \rangle} $$
 - The hidden state $ a^{\langle t \rangle} $ is computed using:
    $$ a^{\langle t \rangle} = \Gamma_o \ast \tanh(c^{\langle t \rangle}) $$

# Bidirectional RNN (BRNN)

 - BRNN is a variant of RNN that allows for information flow in both forward and backward directions within a sequence.
 - It can take into account information from both the past and the future at any point in the sequence.
 - Especially useful in natural language processing (NLP) tasks like named entity recognition (NER).

**BRNN architecture**

 - **Forward RNN:** 
    - Processes the sequence from left to right (past to present).
    - Denoted by $ \overrightarrow{a}^{\langle t \rangle} $ for each time step $ t $.
 - **Backward RNN:** 
    - Processes the sequence from right to left (future to past).
    - Denoted by $ \overleftarrow{a}^{\langle t \rangle} $ for each time step $ t $.
 - Both forward and backward activations are combined to make predictions.

**Computation Process**

 - For a sequence $ x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \dots, x^{\langle t \rangle} $:
    - Compute forward activations: $ \overrightarrow{a}^{\langle 1 \rangle}, \overrightarrow{a}^{\langle 2 \rangle}, \dots, \overrightarrow{a}^{\langle t \rangle} $.
    - Compute backward activations: $ \overleftarrow{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t - 1 \rangle}, \dots, \overleftarrow{a}^{\langle 1 \rangle} $.
    - Combine forward and backward activations at each time step to make predictions:
      $$ \hat{y}^{\langle t \rangle} = g(W_y [\overrightarrow{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t \rangle}] + b_y) $$
      where $ W_y $ is a weight matrix and $ g $ is an activation function (e.g., softmax).

# Deep RNNs

 - A Deep Recurrent Neural Network (Deep RNN) is an extension of a standard RNN where multiple layers of RNN units are stacked on top of each other.
 - This deeper architecture allows the network to learn more complex and abstract representations of the input data, enabling it to capture a hierarchy of temporal patterns in the sequences.
 - For a Deep RNN with $ L $ layers, the hidden state $ \mathbf{a}^{[l] \langle t \rangle} $ at layer $ l $ and time step $ t $ is calculated as follows:
    $$ \mathbf{a}^{[l] \langle t \rangle} = g(W_a^{[l]} [a^{[l]\langle t - 1\rangle}, a^{[l - 1]\langle t \rangle}] + b_a^{[l]}) $$
    Where:
    - $ \mathbf{a}^{[l] \langle t \rangle} $: Activation of the $ l $-th layer at time step $ t $.
    - $ \mathbf{a}^{[l]\langle t - 1 \rangle} $: Activation of the $ l $-th layer at the previous time step $ t-1 $.
    - $ \mathbf{a}^{[l - 1]\langle t \rangle} $: Activation of the previous layer ($ l-1 $) at the current time step $ t $.
    - $ \mathbf{W}_a^{[l]} $: Weight matrix for the $ l $-th layer.
    - $ \mathbf{b}_a^{[l]} $: Bias vector for the $ l $-th layer.
    - $ g $: Activation function (e.g., $ \tanh $, $ \text{ReLU} $).
