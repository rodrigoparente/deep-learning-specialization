# Basic Models

 - Sequence-to-Sequence Models are useful for tasks such as machine translation, speech recognition, and image captioning.
 - The architecture consists of:
    - An encoder that processes the input sequence (e.g., a sentence in the source language) and generates a fixed-size vector representing the entire input.
    - A decoder that takes the vector from the encoder and generates the output sequence (e.g., a translated sentence) one element at a time.
 - Both the encoder and decoder are typically implemented using Recurrent Neural Networks (RNNs).
 - These models require pairs of input-output sequences to learn the mappings effectively.
 - Sequence-to-Sequence improves the quality of generated sequences by exploring multiple possible outputs.

# Picking the Most Likely Sentence

 - Machine translation can be viewed as a conditional language model where the goal is to generate a translation based on an input sentence.
 - Instead of randomly generating translations, the objective is to find the most likely translation that maximizes conditional probability.
 - A greedy search algorithm selects the most likely word at each step based on the current state, but it may lead to suboptimal sequences because it doesn't consider the entire sequence context.
 - Exhaustive search is impractical due to the vast number of potential sentences.
 - The solution is to use a heuristic that explores multiple possible sequences to find the one with the highest overall probability.
 - This heuristic is called Beam Search, and it is more effective in finding optimal translations by considering a broader context of possible sequences.

# Beam Search

 - Beam Search is a heuristic search algorithm used to find the most likely sequence of words in sequence-to-sequence models, such as in machine translation.

**How It Works**
  
 1. **Initialization**: Start with an initial state (e.g., the beginning of a sentence or input sequence).
 2. **Beam Width**: Define a fixed number $ B $, known as the beam width, which determines the number of top candidate sequences to keep at each step.
 3. **Expansion**: 
    - At each step, expand all candidate sequences by adding possible next words.
    - If $ \mathbf{y} = (y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \ldots, y^{\langle T \rangle}) $ represents a sequence of words, and $ \mathbf{x} $ represents the input sequence, the goal is to generate new candidate sequences.
 4. **Scoring**: 
    - Calculate the probability of each expanded sequence:
    $$ P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T_y} P(y^{\langle t \rangle} \mid \mathbf{x}, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}) $$
    Where:
     - $ P(y_t \mid y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}, \mathbf{x}) $ is the probability of the word $ y^{\langle t \rangle} $ given the previous words and the input sequence.
 5. **Pruning**: 
    - Keep only the top $ B $ sequences based on their scores. 
    - The sequences are pruned based on:
     $$ \hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x}) $$
 6. **Iteration**: Repeat the expansion, scoring, and pruning steps until a stopping criterion is met (e.g., reaching a maximum sequence length or generating an end-of-sequence token).
 7. **Output**: The final output is the sequence with the highest score among the top $ B $ sequences.

**Advantages & Limitations**

 - Reduces the number of sequences to consider compared to exhaustive search, making it more computationally feasible.
 - Typically finds a good approximation of the optimal sequence by considering a broader context than greedy search.
 - The quality of the results depends on the chosen beam width. A smaller beam width might miss the optimal sequence, while a larger width increases computational cost.
 - Beam Search does not guarantee finding the absolute best sequence, but rather a good approximation.

# Refinements to Beam Search

 - Longer sentences have lower probabilities because of the multiplication of many probabilities less than one.
 - This makes them less likely to be selected.
 - To avoid issues with numerical underflow, use the log of probabilities:
    $$ P(\mathbf{y} \mid \mathbf{x}) = \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \mid \mathbf{x}, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}) $$
 - Now, to mitigate the bias towards shorter sentences, normalize the score by the sequence length:
    $$ P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \mid \mathbf{x}, y^{\langle 1 \rangle}, \ldots, y^{\langle t-1 \rangle}) $$
    Where:
     - $ T_y $ is the length of the sequence $ y $. 
     - $ \alpha $ is a parameter typically set between 0 (no normalization) and 1 (full normalization).
     - Intermediate values provide a balance between normalization and performance.

**Implementation Considerations**

 - Large beam width consider many possibilities, improving results but with increased computation and memory usage.
 - Small beam width are faster and requires less memory but might result in suboptimal translations.
 - In practice, you should monitor the performance of the beam search and adjust the beam width and normalization parameters as needed to optimize results.

# Error Analysis in Beam Search

 - To determine whether translation errors are due to the beam search algorithm or the underlying RNN model, consider:
    - **Human Translation ( $ y^* $ )**: "Jane visits Africa in September."
    - **Beam Search Output ( $ \hat{y} $ )**: "Jane visited Africa last September."
 - The RNN Model computes $ P(y \mid x) $, the probability of the translation $ y $ given the source sentence $ x $.
 - The Beam Search algorithm tries to find the sequence with the maximum $ P(y \mid x) $ but only keeps track of a fixed number $ B $ of top possibilities.

**Error Analysis Steps**

 1. **Compute Probabilities**: Calculate $ P(y^* \mid x) $ and $ P(\hat{y} \mid x) $ using the RNN model.
 2. **Compare Probabilities**:
    - **Case 1**: $ P(y^* \mid x) > P(\hat{y} \mid x) $
        - Beam search chose $ \hat{y} $, but $ y^* $ has a higher probability.
        - Thus, beam search is failing to find the most likely sequence.
        - In that case, consider increasing the beam width.
    - **Case 2**: $ P(y^* \mid x) \leq P(\hat{y} \mid x) $
        - The RNN model assigns a higher probability to $ \hat{y} $ than $ y^* $.
        - The model may be at fault.
        - Investigate further improvements, such as: 
            - Add regularization.
            - Acquire more training data.
            - Experiment with different network architectures.

# Bleu Score

 - **BLEU (Bilingual Evaluation Understudy)** is a metric used to evaluate the quality of machine-generated translations by comparing them to human-generated references.
 - For example, given the French sentence "Le chat est sur le tapis":
 - Possible human translations might include:
    - "The cat is on the mat."
    - "There is a cat on the mat."
 - A machine translation model might output: "The cat the cat on the mat."
 - The BLEU score provides a single numeric evaluation metric, enabling developers to compare and improve machine translation systems efficiently.
 - It has become a standard measure for evaluating text-generating systems, such as machine translation and image captioning, where multiple valid outputs are possible.

**Final BLEU Score Calculation**

 - The BLEU score is calculated as the geometric mean of the modified precisions for different n-grams, combined using the formula:
  $$ \text{BLEU} = \exp\left(\frac{1}{N}\sum_{n=1}^N \log(P_n) \right) \times \text{BP} $$
  where $ P_n $ is the modified precision for n-grams, and BP is the brevity penalty.

**Brevity Penalty (BP)**

 - BP penalizes translations that are too short, ensuring that machine outputs are not rewarded for being shorter than human references.
 - The formula for BP is:
    - $ BP = 1 $ if the machine translation length exceeds or equals the reference length.
    - $ BP = \exp(1 - \frac{\text{Reference Length}}{\text{Machine Length}}) $ if the machine translation is shorter.

# Attention Model Intuition

 - The Attention Model was introduced as an enhancement to the Encoder-Decoder architecture.
 - It enables the model to focus on specific parts of the input sentence while generating each word of the translation, instead of depending on a single fixed-size context vector.
 - Although originally developed for machine translation, the Attention Model has been successfully applied to various other domains.

**Intuition Behind the Attention Model**

 - The encoder processes the input sentence using a Bidirectional RNN, which computes a rich set of features for each word by considering both the preceding and following words.
 - The decoder generates the translation word by word, focusing on the relevant parts of the input sentence at each step.
 - At each step of the RNN in the decoder, a set of Attention Weights is computed.
 - These weights determine the context vector $ C_t $ used by the RNN to generate the output word at time $ t $.

**Step-by-Step Translation Process with Attention**

 - **First Step ($ S^{\langle 1 \rangle} $):**
    - The first word of the translation is generated by focusing on the first word of the input sentence, guided by attention weights $ \alpha^{\langle 1, 1 \rangle} $, $ \alpha^{\langle 1, 2 \rangle} $, and so on.
  
 - **Second Step ($ S^{\langle 2 \rangle} $):**
    - The second word is generated by focusing on the relevant parts of the input sentence, with a new set of attention weights $ \alpha^{\langle 2, 1 \rangle} $, $ \alpha^{\langle 2, 2 \rangle} $, etc.

 - **Subsequent Steps:**
    - This process continues, with the RNN generating each word based on the previously generated word, the attention context, and the hidden state from the previous step.

# Context and Attention Weights Calculations

 - The Attention Model mimics the human approach to translation by focusing on specific parts of a sentence rather than attempting to memorize the entire sentence.
 - This approach enhances performance, particularly for long sentences, and avoids the performance degradation often seen in traditional Encoder-Decoder architectures.

**Context Calculation**

 - The context vector $ C^{\langle t \rangle} $ is calculated as a weighted sum of the feature vectors:
    $$ C^{\langle t \rangle} = \sum_{t'} \alpha^{\langle t, t' \rangle} \cdot a^{\langle t' \rangle} $$
 - The attention weight $ \alpha^{\langle t, t' \rangle} $ determines how much emphasis is placed on the input word at position $ t' $ when generating the output word at position $ t $.
 - These attention weights are non-negative and sum to one, achieved by applying the softmax function for normalization:
    $$ \alpha^{\langle t, t' \rangle} = \text{softmax}(e^{\langle t, t' \rangle}) $$

**Attention Weights Calculation**

 - The attention scores $ e^{\langle t, t' \rangle} $ are computed using a small neural network, which takes as inputs:
    - $ S^{\langle t-1 \rangle} $: The hidden state from the previous time step.
    - $ a^{\langle t' \rangle} $: The feature vector for the input word at position $ t' $.
 - The score function $ e^{\langle t, t' \rangle} $ is learned through backpropagation during training.
 - Thus, the formula for $ \alpha^{\langle t, t' \rangle} $ is:
    $$ \alpha^{\langle t, t' \rangle} = \dfrac{\exp(e^{\langle t, t' \rangle})}{\sum_{t'=1}^{T_x} \exp(e^{\langle t, t' \rangle})} $$

**Performance and Computational Cost**

 - The Attention Model incurs a quadratic computational cost:
  $$ \text{Cost} = O(T_x \times T_y) $$
  where:
  - $ T_x $ is the length of the input sentence.
  - $ T_y $ is the length of the output sentence.
 - This cost is generally manageable for machine translation tasks, though optimization techniques can further reduce it.
