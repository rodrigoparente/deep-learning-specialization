# Sentiment Classification

- Sentiment classification is the task of determining whether a piece of text expresses positive or negative sentiment.
- It is widely used in Natural Language Processing (NLP) for tasks such as monitoring customer reviews and analyzing social media comments.
- The main challenge of sentiment classification is that labeled training datasets are often limited in size.

**Importance of Word Embeddings**

- Word embeddings enable the creation of effective sentiment classifiers even when labeled training datasets are modest in size.
- Embeddings are vectors that represent words, learned from large text corpora (e.g., billions of words).
- These embeddings help the classifier generalize better to words that might not frequently appear in the labeled training set.

**Basic Sentiment Classification Model**

- Given a sentence (e.g., "The dessert is excellent"), look up each word in a dictionary (e.g., a 10,000-word dictionary).
- Convert words into one-hot vectors. For example, the word "the" might be represented as $\mathbf{0}_{8928}$ in one-hot encoding.
- Multiply the one-hot vectors by an embedding matrix (E), learned from a large corpus, to obtain the embedding vectors.
- Average or sum the embedding vectors to create a 300-dimensional feature vector:
  $$ \mathbf{v} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{w}_i $$ 
  Where:
  - $n$ is the number of words.
  - $\mathbf{w}_i$ are the word vectors.
- Pass the feature vector into a softmax classifier to predict the sentiment ($\hat{y}$).

**Limitations of the Basic Model**

- The basic model ignores word order, which can lead to incorrect sentiment predictions.
- For example, a review stating "Completely lacking in good taste, good service, and good ambiance" might be incorrectly classified as positive due to the frequent occurrence of the word "good."
- This limitation can be addressed by using a Recurrent Neural Network (RNN), which can account for word order and thus correctly interpret phrases like "not good" as negative.

**Improved Sentiment Classification Model Using RNN**

- Convert the text into one-hot vectors and then into word embeddings using matrix E.
- Feed the sequence of embeddings into an RNN.
- Use the final hidden state of the RNN to predict the sentiment ($\hat{y}$).
- The RNN can capture the sequence of words and their context, enabling more accurate sentiment prediction.

# Debiasing Word Embeddings

 - Machine Learning (ML) and Artificial Intelligence (AI) algorithms are increasingly used to make important decisions in areas such as college admissions, job searches, loan applications, and criminal justice.
 - Word embeddings can reflect societal biases present in the text used for training.
 - For example, some ML models may exhibit gender stereotypes by associating "man:computer programmer" with "woman:homemaker" and "father:doctor" with "mother:nurse."
 - To ensure fairness in these algorithms, we must eliminate undesirable biases such as those related to gender, ethnicity, sexual orientation, and socioeconomic status.

**Identifying Bias in Word Embeddings**

 - The first step in reducing bias in word embeddings is identifying the direction that corresponds to a particular bias.
 - For gender bias, this can be done by subtracting the embedding vector for "she" from "he" and similarly for other gendered word pairs (e.g., male - female).
 - The result reveals a direction in the embedding space that corresponds to gender bias.
 - This bias direction is orthogonal to the subspace unrelated to the bias.

**Neutralization of Bias**

 - For words that are not gender-specific (e.g., "doctor," "babysitter"), the goal is to reduce or eliminate their component in the bias direction.
 - Words that inherently reflect gender (e.g., "grandmother," "grandfather") are excluded from neutralization because gender is part of their definition.

**Equalization Process**

 - Equalization involves adjusting word pairs in the embedding space so they maintain the same distance from neutral words.
 - This process ensures that pairs of words (e.g., "grandmother" and "grandfather") are equidistant from gender-neutral words (e.g., "babysitter," "doctor").
 - This prevents the reinforcement of harmful stereotypes, such as associating "grandmother" more closely with "babysitter" than "grandfather."

**Classifier for Gender-Specific Words**

 - A classifier can be used to automatically determine which words should be gender-specific (e.g., "grandmother," "grandfather") and which should not (e.g., "doctor").
 - Most words in the English language are not gender-specific, meaning they do not inherently carry gender, ethnicity, or other biases.
 