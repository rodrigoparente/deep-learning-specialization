# Programming Frameworks 

 - Implementing deep learning algorithms from scratch using Python and NumPy is a valuable learning experience.
 - For more complex models (e.g., Convolutional Neural Networks, Recurrent Neural Networks) or large-scale models, implementing everything from scratch becomes impractical.
 - Just as it's more efficient to use optimized libraries for matrix multiplication, deep learning frameworks streamline the implementation of complex models.
 - The choice of framework may depend on factors such as preferred programming language (e.g., Python, Java, C++) and the specific application domain (e.g., computer vision, natural language processing, online advertising).

**Criteria for Choosing a Framework**

 1. **Ease of Programming:**
     - Includes ease of developing and iterating on neural networks.
     - Considerations for deployment to production, handling large-scale applications.
 2. **Running Speeds:**
     - Efficiency in training models on large datasets.
     - Performance may vary between frameworks.
 3. **Openness and Governance:**
     - True openness involves being open source and having good governance.
     - Be cautious of frameworks that may shift from open source to proprietary or cloud-based services over time.

# Tensorflow

 - TensorFlow is an open-source framework developed by Google for designing, training, and deploying machine learning and deep learning models.
 - It provides a comprehensive ecosystem that includes tools and libraries for building machine learning models, optimizing performance, and deploying models on various platforms.
 - Key Features:
    - **Flexibility:** Supports both high-level APIs (e.g., Keras) and low-level APIs for detailed control.
    - **Scalability:** Runs on various platforms, including CPUs, GPUs, and TPUs.
    - **Ecosystem:** Includes TensorFlow Serving for production, TensorFlow Lite for mobile, and TensorFlow Extended (TFX) for end-to-end ML pipelines.

**Simple NN using TensorFlow**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a simple neural network model
model = Sequential([
    Dense(4, activation='relu'),    # Input layer with 4 neurons one for each feature
    Dense(3, activation='softmax')  # Output layer with 3 neurons (one for each class)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=5, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Make predictions on new data
predictions = model.predict(X_test)
print("Predictions:", predictions)
```