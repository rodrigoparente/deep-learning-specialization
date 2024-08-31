# Data Splitting for Training

 - In the traditional approach, data is typically split into training, development (dev), and test sets.
 - Traditional ratioswould be 70/30 for train/test or 60/20/20 for train/dev/test.
 - In the era of big data, dev and test sets are often smaller percentages.
 - With a million data points, a split might be 98% train, 1% dev, 1% test.
 
**Good Practices**

 - Ensure that dev and test sets come from the same distribution, even if the training set differs.
 - When a test set is not available, the dev set can serve for both validation and final evaluation, but caution is needed to avoid overfitting.
 - A well-structured train/dev/test split allows for quicker iteration and more effective measurement of bias and variance, aiding in the selection of the best model architecture.

# Bias vs. Variance

 - Bias refers to underfitting, where the model is too simple to capture the underlying data pattern, resulting in poor performance even on the training set.
 - Variance refers to overfitting, where the model is too complex and fits the training data too closely, failing to generalize well to new data.

By analyzing the errors on the training and development (dev) sets, you can diagnose whether a model has high bias, high variance, or both.

 - High bias means the model underfits the training data, while high variance indicates it doesn't generalize well to the dev set.
 - In practice, the error on the training set reveals bias issues, while the difference in errors between the training and dev sets indicates variance problems.
 - A balanced model will have low bias and low variance.


# Solving Bias and Variance Problem

To systematically improve a neural network's performance, start by diagnosing whether the algorithm has high bias or high variance:

1. **High Bias:** If the model underfits the training data, you can:
   - Increase the network size (more hidden layers or units).
   - Train longer or use advanced optimization techniques.
   - Experiment with different network architectures.

2. **High Variance:** If the model overfits the training data but performs poorly on the dev set, you can:
   - Acquire more data.
   - Apply regularization techniques.
   - Experiment with different network architectures.
