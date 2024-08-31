# Softmax Regression

 - Softmax regression, also known as multinomial logistic regression, is a generalization of logistic regression used for multiclass classification problems.
 - It is applied when the goal is to classify an input into one of $ C $ different classes, where $ C > 2 $.
 - The model outputs a probability distribution vector $ \hat{y} $ across all $ C $ classes, with dimensions $ (C, 1) $.
 - The sum of the probabilities in $ \hat{y} $ for all classes is 1.

**Softmax Activation Function**

 1. Compute the logits for the final layer $ L $ using:
    $$ z_L = W_L a_{L-1} + b_L $$
    where:
     - $ W_L $ is the weight matrix of the layer $ L $.
     - $ a_{L-1} $ is the activation from the previous layer
     - $ b_L $ is the bias of the layer $ L $.
 2. Calculate $ t = e^{z_L} $ element-wise, where $ z_L $ is the vector of logits (pre-activation values).
 3. Normalize $ t $ to obtain the output vector $ a_L $ (which is also $ \hat{y} $):
    $$ a_{L,i} = \frac{e^{z_{L,i}}}{\sum_{j=1}^{C} e^{z_{L,j}}} $$
    where:
    - $ i $ is an index that represents a specific class in the output vector $ a_L $.