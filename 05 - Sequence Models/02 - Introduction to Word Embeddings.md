# Word Representation

 - Traditionally, words are represented using one-hot vectors, where each word corresponds to a unique index in a vocabulary.
 - One-hot encoding treats each word as independent, making it challenging for algorithms to generalize across similar words (e.g., "apple" and "orange").
 - Word embedding is a method of converting words into vectors of numbers.
 - Embeddings capture semantic relationships, enabling algorithms to understand analogies, such as "man is to woman as king is to queen."
 - Word embedding facilitate the development of NLP applications, even with relatively small labeled training sets.
 - The term "embedding" refers to the process of mapping words to points in a high-dimensional space.

**Advantages of Word Embeddings**

 - Instead of one-hot vectors, words can be represented as feature vectors that capture various attributes (e.g., gender, royalty, age, food).
 - This approach allows for better generalization; for example, "orange" and "apple" will have similar embeddings, making it easier for an algorithm to recognize related phrases like "orange juice" and "apple juice."

**Visualization of Word Embeddings**

 - The *t-SNE algorithm* is a common technique for visualizing high-dimensional word embeddings by projecting them into a 2D space.
 - Similar words (e.g., "man" and "woman," "king" and "queen," as well as categories like fruits, animals, and numbers) tend to cluster together in these visualizations.

# Using Word Embeddings

 - Word embeddings are learned from large, unlabeled text corpora (e.g., billions of words from the Internet).
 - These embeddings capture relationships (e.g., "orange" and "durian" are both fruits) and can be transferred to tasks with smaller labeled datasets.

**Transfer Learning**

 - **Step 1:** Learn word embeddings from a large corpus (or use pre-trained embeddings available online).
 - **Step 2:** Transfer the learned embeddings to a new task (e.g., NER) with a smaller labeled training set.
 - **Step 3 (Optional):** Continue fine-tuning embeddings with new data if the labeled dataset is large enough.

Replacing one-hot vectors with embedding vectors improves generalization and reduces the need for large labeled datasets.

**Comparison with Face Recognition**

 - Both involve learning vector representations (e.g., 128-dimensional for faces, 300-dimensional for words).
 - But, face Recognition, the network computes encodings for any new face.
 - For word embeddings, a fixed embeddings are learned for a vocabulary of words (e.g., 10,000 words).

# Properties of Word Embeddings

 - Word embeddings capture relationships between words and can be used for analogy reasoning.
 - Analogy reasoning, though not the most critical NLP application, provides insight into what word embeddings can achieve.
 - For example, consider the analogy: "man is to woman as king is to what?"
 - A common answer would be: "man is to woman as king is to queen."
 - The algorithm can automatically figure this out by analyzing word embeddings.

**Representation of Words in Word Embeddings**

 - Each word is represented as a vector (e.g., $ \textbf{e}_{\text{man}}, \textbf{e}_{\text{woman}}, \textbf{e}_{\text{king}}, \textbf{e}_{\text{queen}} $) in a multi-dimensional space.
 - By subtracting the vectors of "man" and "woman" ($ \textbf{e}_{\text{man}} - \textbf{e}_{\text{woman}} $), you get a vector that represents the gender difference.
 - Similarly, the difference between "king" and "queen" vectors ($ \textbf{e}_{\text{king}} - \textbf{e}_{\text{queen}} $) captures the gender difference.
 - These differences are approximately the same, which is why the analogy holds true.
 - To solve the analogy, the algorithm computes $ \textbf{e}_{\text{man}} - \textbf{e}_{\text{woman}} $ and looks for a word vector $ \textbf{e}_{w} $ such that:
    $$ \textbf{e}_{w} \approx \textbf{e}_{\text{king}} - \textbf{e}_{\text{man}} + \textbf{e}_{\text{woman}} $$
  - The word $ w $ that maximizes this similarity is chosen, and in this case, $ w $ would be "queen."

**Cosine Similarity as a Similarity Measure**

 - You can use different measures to calculate the similarity of two embeddings, for example:
    - **Cosine Similarity:**
        - Defined as $ \text{similarity}(\textbf{u}, \textbf{v}) = \frac{\textbf{u} \cdot \textbf{v}}{||\textbf{u}|| \times ||\textbf{v}||} $.
        - Measures the cosine of the angle between two vectors $ \textbf{u} $ and $ \textbf{v} $.
        - If the angle $ \theta $ between the vectors is 0, cosine similarity is 1; if $ \theta = 90^\circ $, cosine similarity is 0; and if $ \theta = 180^\circ $, cosine similarity is -1.
    - **Euclidean Distance:**
        - Euclidean distance ($ ||\textbf{u} - \textbf{v}||^2 $) can also be used as a dissimilarity measure, where the negative of this value is considered for similarity.

# Embedding Matrix

 - An embedding matrix is a key component in natural language processing (NLP) models, particularly those involving word embeddings.
 - It is a matrix used to store the vector representations (embeddings) of words in a continuous vector space.
 - Each row of the embedding matrix corresponds to a word in the vocabulary, and the values in that row represent the coordinates of the word in the embedding space.

**Embedding Matrix Structure**

 - The embedding matrix $ \mathbf{E} $ is typically of size $ |V| \times d $, where $ |V| $ is the size of the vocabulary (the number of unique words) and $ d $ is the dimensionality of the embedding space.
 - Each row in the matrix corresponds to a word, and each column corresponds to a dimension in the embedding space.
    $$
        \mathbf{E} =
        \begin{pmatrix}
        \mathbf{e}_1 \\
        \mathbf{e}_2 \\
        \vdots \\
        \mathbf{e}_{|V|}
        \end{pmatrix}
    $$
    Where, $ \mathbf{e}_i $ is the embedding vector for the $ i $-th word in the vocabulary.