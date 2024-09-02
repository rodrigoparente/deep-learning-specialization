# Speech Recognition

 - Speech recognition is technology that converts spoken language into text or commands.
 - It involves processing audio signals, extracting features, and matching them to words based on context and language patterns.
 - Applications include voice assistants and transcription services.

**Connectionist Temporal Classification (CTC)**

 - Connectionist Temporal Classification (CTC) was developed by Alex Graves, Santiago Fernandez, Faustino Gomez, and Jürgen Schmidhuber.
 - CTC addresses the issue of varying lengths between audio input and text output by allowing the model to produce sequences with repeated characters and special "blank" characters.
 - Blank characters indicate no output and help align variable-length input sequences with shorter output sequences.

**Loss Function**

 - For instance, if an audio clip says "the quick brown fox," the model might output "ttt_eee___ qqq__u__ii__cccc___kkk" (where "_" is the blank character).
 - The CTC loss function collapses repeated characters not separated by blanks into a single character, resulting in a final output of "the quick."

# Trigger Word Detection

 - These systems enable devices to respond to specific spoken words or phrases.
 - Examples include:
    - **Amazon Echo:** Activated by "Alexa."
    - **Baidu DuerOS:** Activated by "xiaodunihao."
    - **Apple Siri:** Activated by "Hey Siri."
    - **Google Home:** Activated by "Okay Google."

**Building a Trigger Word Detection System**
 
 - Start with an audio clip and compute its spectrogram features.
 - Use a Recurrent Neural Network (RNN) to process these features.
 - Label audio segments where the trigger word is detected with a $ 1 $, and $ 0 $ when is not.

**Imbalanced Training Set**

 - raining sets often have many more $ 0 $ (no trigger word) than $ 1 $ (trigger word present), leading to imbalance.
 - To address this problem, label several consecutive time steps around the trigger word with $ 1 $ instead of just one time step.
 - This increases the number of positive labels, balancing the ratio of $ 1 $ to $ 0 $, and improves model training by providing more positive examples.