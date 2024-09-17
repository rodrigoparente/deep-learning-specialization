Here are the questions and answers from the images converted to markdown format:

1. A Transformer Network, like its predecessors RNNs, GRUs and LSTMs, can process information one word at a time. (Sequential architecture).

 - [x] False
 - [ ] True

   Correct: A Transformer Network can ingest entire sentences all at the same time.

2. The major innovation of the transformer architecture is combining the use of LSTMs and RNN sequential processing.

 - [x] False
 - [ ] True

   Correct: The major innovation of the transformer architecture is combining the use of attention based representations and a CNN convolutional neural network style of processing.

3. What are the key inputs to computing the attention value for each word?

 - [ ] The key inputs to computing the attention value for each word are called query, knowledge, and vector.
 - [ ] The key inputs to computing the attention value for each word are called quotation, knowledge, and value.
 - [x] The key inputs to computing the attention value for each word are called query, key, and value.
 - [ ] The key inputs to computing the attention value for each word are called quotation, key, and vector.

   Correct: The key inputs to computing the attention value for each word are called query, key, and value.


4. Which of the following correctly represents Attention?

 - [x] $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
 - [ ] $Attention(Q, K, V) = min(\frac{QK^T}{\sqrt{d_k}})V$
 - [ ] $Attention(Q, K, V) = min(\frac{QK^T}{-})V$

5. Which of the following statements represents Key (K) as used in the self-attention calculation?

 - [ ] K = interesting questions about the words in a sentence
 - [ ] K = the order of the words in a sentence
 - [x] K = qualities of words given a Q
 - [ ] K = specific representations of words given a Q

   Correct: The qualities of words given a Q are represented by Key (K).

6. $Attention(W_i^QQ, W_i^KK, W_i^VV)$

   i here represents the computed attention weight matrix associated with the ith "word" in a sentence.

 - [ ] True
 - [x] False

   Correct: i here represents the computed attention weight matrix associated with the ith "head" (sequence).

7. What information does the Decoder take from the Encoder for its second block of Multi-Head Attention?

 - [ ] Q
 - [x] K
 - [x] V

   Correct: Great, you got all the right answers.

8. The output of the decoder block contains a softmax layer followed by a linear layer to predict the next word one word at a time?

 - [x] False
 - [ ] True

   Correct: The output of the decoder block contains a linear layer followed by a softmax layer to predict the next word one word at a time.

9. Why is positional encoding important in the translation process? (Check all that apply)

 - [x] Position and word order are essential in sentence construction of any language.
 - [ ] It helps to locate every word within a sentence.
 - [ ] It is used in CNN and works well there.
 - [x] Providing extra information to our model.

   Correct: Great, you got all the right answers.

10. Which of these is a good criterion for a good positional encoding algorithm?

 - [x] It should output a unique encoding for each time-step (word's position in a sentence).
 - [x] Distance between any two time-steps should be consistent for all sentence lengths.
 - [x] The algorithm should be able to generalize to longer sentences.
 - [ ] None of these.

   Correct: Great, you got all the right answers.