# Why Human-level Performance?

 - Advances in deep learning have made ML systems increasingly competitive with human-level performance.
 - Designing ML systems is more efficient when focusing on tasks humans excel at, making comparisons to human performance natural.

**Performance Trends**

 - ML systems improve rapidly as they approach human-level performance.
 - After surpassing human-level performance, progress slows as the system nears the theoretical limit, known as Bayes optimal error.
 - Bayes Error represents the lowest possible error, often close to human-level performance, leaving little room for further improvement.
 - Techniques that are useful when ML systems are below human-level performance include:
    - Hiring humans to label data, providing more training examples for the model.
    - Analyzing errors to understand why humans succeed where the algorithm fails.
    - Tuning the model based on insights from human performance.
 - These techniques become less effective once the model surpasses human capabilities.

# Avoidable Bias

 - Avoidable Bias is the difference between the training error and the Bayes error (or an estimate of it).
 - When the training error is high relative to the Bayes error, this means there's potential to improve the model by reducing bias.
 - For example, if human-level error is $1\%$ and your model's training error is $8\%$, the avoidable bias is $7\%$. 
 - This suggests substantial room for improving the model to reduce bias.
 - To address avoidable bias, focus on strategies that reduce model bias, such as:
    - Training a more complex model (e.g., a larger neural network).
    - Extending training time or using more training data.

# Understanding Human-level Performance

 - Human-level performance is the error rate achieved by humans on a given task.
 - It often serves as a benchmark to estimate Bayes error, which is the lowest achievable error rate for that task.
 - The best performance observed from experts or a group of experts provides an upper limit for Bayes error.
 - In research, use the highest human performance as an estimate of Bayes error to evaluate how close your machine learning model is to the theoretical limit.
 - For deployment, surpassing the performance of an average or typical human may be sufficient.
 - When analyzing bias versus variance:
    - If your model’s error is significantly higher than human-level performance, focus on reducing bias.
    - If your model’s error is close to but still higher than human-level performance, focus on reducing variance.

# In Summary

 - For improving model performance you have to:
    - Fit the training set well (low avoidable bias).
    - Ensure good generalization from training to dev/test set (low variance).
 - To accomplish that, you can:
    - Use tactics to reduce **avoidable bias**, such as:
      - Training a Bigger Model: Increase model capacity.
      - Training Longer: Extend training duration.
      - Better Optimization Algorithms: Use algorithms like Adam, RMSprop, or ADS momentum.
      - Improved Architecture: Experiment with different neural network architectures and hyperparameters, including activation functions, layer numbers, and units.
    - Address **variance** issues with strategies like:
      - More Data: Acquire additional training data to improve generalization.
      - Regularization: Apply techniques such as L2 regularization, dropout, and data augmentation.
      - Architecture and Hyperparameters: Continue experimenting with neural network designs and hyperparameters to better fit the problem.