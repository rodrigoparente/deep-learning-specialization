# Transformer Network Intuition

 - Transformers have revolutionized Natural Language Processing (NLP) with their advanced architecture.
 - They are more complex than RNNs, GRUs, and LSTMs, but they offer significant improvements.
 - Unlike RNNs, transformers process entire sequences in parallel, allowing simultaneous computation for all tokens.

**Core Innovations**

 - Transformers combine attention-based representations with CNN-style parallel processing.
 - Key components include:
    - **Self-Attention:** Computes multiple rich representations for each token in parallel.
    - **Multi-Headed Attention:** Enhances self-attention by creating multiple versions of these representations for richer information.

# Self-Attention

 - Self-attention allows transformers to compute attention-based representations for each word in a sentence.
 - For a sentence like "Jane visite l'Afrique en septembre," self-attention computes five representations (one for each word).
  
**Query, Key, and Value ($ Q $, $ K $, $ V $)**

 - Words are initially represented by embeddings, but self-attention refines these representations based on context.
 - Each word is associated with three vectors: query ($ Q $), key ($ K $), and value ($ V $).
 - These vectors are derived from the word embedding using learned matrices $W^{\langle Q \rangle}$, $W^{\langle K \rangle}$, and $W^{\langle V \rangle}$.

**Self-Attention Calculation**

 1. Compute Q, K, and V vectors:
       - $ Q^{\langle t \rangle} = W^Q \cdot X^{\langle t \rangle} $
       - $ K^{\langle t \rangle} = W^K \cdot X^{\langle t \rangle} $
       - $ V^{\langle t \rangle} = W^V \cdot X^{\langle t \rangle} $
 2. Compute attention scores by taking the dot product of the query vector $Q_3$ with each key vector $K_i$.
 3. Apply the softmax function to these scores to get attention weights.
 4. Multiply the attention weights by the value vectors $V_i$ and sum them to get the final representation for "l'Afrique."

**Mathematical Representation**

 - The attention value for a word is computed as:
    $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$
    where $d_k$ is the dimensionality of the key vectors.
 - This scaled dot-product attention mechanism helps to prevent the dot-product values from becoming too large.

**Advantages of Self-Attention**

 - Generates contextually relevant representations for each word.
 - Allows the model to understand relationships between words regardless of their position in the sentence.
 - Results in richer and more nuanced representations than static word embeddings.

# Multi-Head Attention

 - Multi-head attention extends self-attention by performing the attention mechanism multiple times in parallel.
 - Each individual computation of self-attention is called a "head."

**Multi-Head Attention Formula**

 - For each head $ i $ computes:
    $$ \text{Attention}_i(Q, K, V) = \text{Softmax} \left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i $$
    where:
     - $ Q_i $, $ K_i $, and $ V_i $ are the query, key, and value matrices for head $ i $.
     - $ d_k $ is the dimension of the key vectors.
 - The final multi-head attention output is:
    $$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{Attention}_1, \text{Attention}_2, \ldots, \text{Attention}_H) W_O $$
    where:
     - $ H $ is the number of heads.
     - $ W_O $ is a learned output weight matrix.

**Parallel Computation**

 - While conceptually done in sequence, heads are often computed in parallel during implementation because they are independent of each other.

# Transformer Network

 - Transform sequences from one domain to another (e.g., translation, text generation).
 - Consists of an encoder and a decoder, each comprising multiple layers.

**Encoder Block**

 - Takes as input a sequence embeddings, including positional encodings.
 - Computes attention scores using Query (Q), Key (K), and Value (V) vectors:
    $$ \text{Attention}(Q, K, V) = \text{Softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$
 - It's responsible for extract features from the attention output.
  
**Decoder Block**

  - Begins with a start token and incorporates previously generated tokens.
  - First block uses output from the encoder and previous tokens.
  - Second block computes attention with encoder outputs to focus on relevant parts of the source sequence.
  - It's responsible for predicting the next token in the sequence based on the attention outputs.

**Positional Encoding**

 - Adds information about the position of tokens within the sequence.
 - Uses sine and cosine functions to create unique position embeddings for each token.
 - Formula for positional encoding:
    $$ \text{PE}_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/D}} \right) $$
    $$ \text{PE}_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/D}} \right) $$