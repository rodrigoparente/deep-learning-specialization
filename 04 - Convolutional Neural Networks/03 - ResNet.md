# ResNet

 - Very deep neural networks often suffer from vanishing and exploding gradient problems, making them difficult to train effectively.
 - Skip connections allow activations from one layer to be fed directly to a much deeper layer in the network, bypassing intermediate layers.
 - ResNet (Residual Network) leverages skip connections, enabling the training of networks with over 100 layers by mitigating the vanishing/exploding gradient issue.
  
**Residual Blocks** 

 - Residual blocks are the building blocks of ResNet. 
 - Consider two consecutive layers: from activation $ a^{[l]} $ to $ a^{[l+2]} $.
    - Traditional path: 
        - Compute $ z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]} $, followed by ReLU to get $ a^{[l+1]} $.
        - Similarly, compute $ z^{[l+2]} $ and apply ReLU to get $ a^{[l+2]} $.
    - Residual block: 
        - Instead of just relying on the main path, $ a^{[l]} $ is directly added to $ z^{[l+2]} $ before the ReLU activation, resulting in $ a^{[l+2]} = \text{ReLU}(z^{[l+2]} + a^{[l]}) $.
    - This forms a shortcut connection, which bypasses the intermediate layers and directly adds the earlier activation to the output.
  
**Benefits of Residual Networks**

 - The shortcut connection helps preserve the gradient flow, making it easier to train very deep networks.
 - Stacking multiple residual blocks allows for the construction of deep networks, such as ResNets, with improved performance.
 - Empirically, without residual blocks, deeper networks show higher training error after a certain depth, contrary to theoretical expectations.
 - ResNets, however, demonstrate reduced training error even as depth increases, effectively handling networks with over 100 layers.
  
**Training Dynamics**

 - Plain networks (without residual connections) tend to struggle with deeper architectures, showing increased training error.
 - ResNets maintain or even reduce training error as the network depth increases, sometimes extending to networks with over 1000 layers.

# Why ResNets Work Well

 - Consider a neural network with input $ X $ leading to some activation $ a^{[l]} $.
 - Suppose you modify the network by adding two more layers, creating a deeper network with activation $ a^{[l+2]} $.
 - In a plain network (without residual connections), adding these layers could hurt the network’s performance.
 - In a ResNet block, however, a skip connection is added from $ a^{[l]} $ to $ a^{[l+2]} $, creating a direct shortcut.
 - The output is then $ a^{[l+2]} = \text{ReLU}(z^{[l+2]} + a^{[l]}) $, where $ z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]} $.

**Learning the Identity Function**

 - If the weight matrix $ W^{[l+2]} $ and bias $ b^{[l+2]} $ are small or zero (e.g., due to $ L_2 $ regularization), the residual block can easily learn the identity function: $ a^{[l+2]} = a^{[l]} $.
 - This makes it easier for the network to learn, as it doesn’t degrade the performance by merely adding layers.
 - The identity function being easy to learn ensures that adding residual blocks won't hurt performance and may even improve it by learning more complex representations.