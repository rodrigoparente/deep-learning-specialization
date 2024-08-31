# Why ML Strategy

 - To illustrate the importance of strategy, consider a scenario where you're working on a cat classification project and have achieved 90% accuracy.
 - Nevertheless, this performance it's not enough for your application.
 - At this point, you might consider various approaches to improve the system, such as:
    - Collecting more training data.
    - Enhancing the diversity of your dataset.
    - Experimenting with different optimization algorithms like Adam.
    - Adjusting your network architecture.
 - However, choosing the wrong approach could result in wasted time and effort, as some teams have spent months on ineffective strategies.

# Orthogonalization

 - In the context of machine learning, orthogonalization refers to the process of decoupling different aspects of a model's performance so that improvements in one area can be made independently of others.
 - This approach helps streamline the development and optimization process by ensuring that efforts to improve one part of the model do not inadvertently affect or complicate other aspects.

For example, consider a machine learning pipeline where you want to improve both the model's accuracy and its ability to generalize to new data. By applying orthogonalization, you might separately focus on:

 1. **Improving training accuracy**: This could involve optimizing the model architecture or fine-tuning hyperparameters.
 2. **Improving generalization**: This might include techniques like regularization, data augmentation, or cross-validation.

Orthogonalization ensures that each of these goals can be addressed independently, leading to a more efficient and targeted optimization process. The idea is to create "orthogonal" (independent) tasks, where solving one does not interfere with solving the other, allowing for a more systematic approach to improving machine learning systems.