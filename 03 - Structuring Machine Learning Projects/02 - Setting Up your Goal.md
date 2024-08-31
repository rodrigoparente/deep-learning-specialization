# Single Number Evaluation Metric

 - Machine learning is **empirical**, involving a cycle of idea formulation, coding, experimentation, and refinement.
 - Tuning hyperparameters or testing different algorithms becomes more efficient with a single real number evaluation metric.
 - For any machine learning project, it's beneficial to establish a single real number evaluation metric from the start.
 - For example, in a cat classification project, after training multiple classifiers, an effective way to compare them could be to use the f1-score metric.
 - A single evaluation metric, combined with a well-defined development set, accelerates the iterative process of improving machine learning algorithms.

# Satisficing and Optimizing Metric

 - It's not always straightforward to combine all desired metrics into a single evaluation metric.
 - In such cases, it can be useful to set up **optimizing** and **satisficing** metrics.
 - The optimizing metric is the one you want to maximize and the satisficing metrics are the constraints.
 - For example, suppose you care about the classification accuracy and the running time of a classifier.
 - You can choose a classifier that maximizes accuracy (the optimizing metric) while ensuring the running time (the satisficing metric) is less than or equal to a specific threshold (e.g., 100ms).
 - This approach provides a clear way to select the best model or classifier.

# Train/Dev/Test Distributions

 - Properly setting up training, development (dev), and test sets is essential for efficient progress in machine learning projects.
 - The dev set is used to evaluate and select the best model during development, while the test set is for final evaluation.
 - If dev and test sets are from different distributions, months of optimization on the dev set may result in poor performance on the test set, leading to wasted effort.
 - To avoid this, ensure that both dev and test sets come from the same distribution by randomly shuffling and splitting data.

# Size of the Dev and Test Sets

 - Traditional rules of thumb, such as splitting data 70/30 for train and test, or 60/20/20 for train, dev, and test, were suitable when datasets were smaller.
 - In the current Deep Learning era, with much larger datasets, these traditional splits are often outdated.
 - With large datasets (e.g., millions of examples), it’s common to allocate a much larger portion (e.g., 98%) of the data to the training set.
 - Dev and test sets can be significantly smaller than before, sometimes as little as 1% of the total data, which might still be sufficient.
 - In some scenarios, especially when high confidence in the final system isn’t necessary, it may be acceptable to only have a train and dev set.

# When to Change the Dev/Test Sets and Metrics

 - The dev set and evaluation metric guide the project, acting as a target for the team.
 - If at some point, you realize the target is misplaced during the project, it should be adjusted.
 - For example, you build a classifier with classification error as the chosen metric:
    - Algorithm A has a 3% error rate but lets through inappropriate content.
    - Algorithm B has a 5% error rate but avoids inappropriate content.
 - The metric suggests Algorithm A is better, but in reality, Algorithm B aligns more with user and company needs.
 - To address this, a weighting factor can be introduced in the error formula to penalize critical misclassifications more heavily.
    $$ \text{Weighted Error} = \frac{1}{\sum_{i} w(i)} \sum_{i=1}^{m} w(i) \cdot \left(\hat{y}^{(i)} \neq y^{(i)}\right) $$
    Where:
     - $w_i$ is a term to penalize certain types of misclassifications more heavily.
 - This makes the metric better reflect the application's needs.
