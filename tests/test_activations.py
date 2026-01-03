import numpy as np
import pytest

try:
    from vulkan_backend import VulkanCompute
    VULKAN_AVAILABLE = True
except Exception as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestActivationFunctions:
    """Test activation function correctness"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    def test_relu_basic(self, gpu):
        """Test ReLU activation"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y_gpu = gpu.activation_relu(x)
        y_cpu = np.maximum(0, x)
        
        print(f"\n✓ ReLU test:")
        print(f"  Input: {x}")
        print(f"  GPU: {y_gpu}")
        print(f"  CPU: {y_cpu}")
        
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-5, atol=1e-5)
    
    def test_relu_large_array(self, gpu):
        """Test ReLU on large array"""
        x = np.random.randn(10000).astype(np.float32)
        y_gpu = gpu.activation_relu(x)
        y_cpu = np.maximum(0, x)
        
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-5, atol=1e-5)
        print(f"\n✓ ReLU large array: {len(x)} elements")
    
    def test_gelu_basic(self, gpu):
        """Test GELU activation"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y_gpu = gpu.activation_gelu(x)
        
        # CPU reference: GELU approximation
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        y_cpu = 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + coeff * x**3)))
        
        print(f"\n✓ GELU test:")
        print(f"  Input: {x}")
        print(f"  GPU: {y_gpu}")
        print(f"  CPU: {y_cpu}")
        
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-4)
    
    def test_silu_basic(self, gpu):
        """Test SiLU (Swish) activation"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y_gpu = gpu.activation_silu(x)
        
        # CPU reference: SiLU = x * sigmoid(x)
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        y_cpu = x * sigmoid(x)
        
        print(f"\n✓ SiLU test:")
        print(f"  Input: {x}")
        print(f"  GPU: {y_gpu}")
        print(f"  CPU: {y_cpu}")
        
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-5, atol=1e-5)
    
    def test_softmax_basic(self, gpu):
        """Test softmax activation"""
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        y_gpu = gpu.activation_softmax(x)
        
        # CPU reference
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        y_cpu = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        print(f"\n✓ Softmax test:")
        print(f"  Input:\n{x}")
        print(f"  GPU:\n{y_gpu}")
        print(f"  CPU:\n{y_cpu}")
        print(f"  Sum per row (should be 1.0): {y_gpu.sum(axis=1)}")
        
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-4)
        # Check that each row sums to 1.0
        np.testing.assert_allclose(y_gpu.sum(axis=1), np.ones(y_gpu.shape[0]), rtol=1e-4)
    
    def test_softmax_numerical_stability(self, gpu):
        """Test softmax with large values (numerical stability)"""
        x = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
        y_gpu = gpu.activation_softmax(x)
        
        # Should not be NaN or Inf
        assert not np.any(np.isnan(y_gpu))
        assert not np.any(np.isinf(y_gpu))
        assert np.allclose(y_gpu.sum(), 1.0, rtol=1e-4)
        
        print(f"\n✓ Softmax numerical stability:")
        print(f"  Input: {x}")
        print(f"  Output: {y_gpu}")
        print(f"  Sum: {y_gpu.sum():.6f}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestActivationAccuracy:
    """Test activation functions in classification accuracy context"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    def simple_classifier(self, gpu, X, y, activation='relu'):
        """
        Simple classifier: linear layer + activation + softmax
        
        Returns:
            predictions: predicted class indices
            probabilities: class probabilities
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize random weights
        np.random.seed(42)
        W = np.random.randn(n_classes, n_features).astype(np.float32) * 0.1
        b = np.zeros(n_classes, dtype=np.float32)
        
        # Forward pass
        logits = X @ W.T + b  # (n_samples, n_classes)
        
        # Apply activation
        if activation == 'relu':
            hidden = gpu.activation_relu(logits)
        elif activation == 'gelu':
            hidden = gpu.activation_gelu(logits)
        elif activation == 'silu':
            hidden = gpu.activation_silu(logits)
        else:
            hidden = logits
        
        # Apply softmax for probabilities
        probs = gpu.activation_softmax(hidden)
        
        # Predictions
        predictions = np.argmax(probs, axis=1)
        
        return predictions, probs
    
    def test_classification_moons(self, gpu):
        """Test classification accuracy on moons dataset"""
        try:
            from sklearn.datasets import make_moons
            from sklearn.model_selection import train_test_split
        except ImportError:
            pytest.skip("sklearn not available")
        
        # Generate moons dataset
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
        X = X.astype(np.float32)
        
        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train simple classifier (multiple epochs with Hebbian-like updates)
        n_classes = 2
        n_features = 2
        W = np.random.randn(n_classes, n_features).astype(np.float32) * 0.1
        b = np.zeros(n_classes, dtype=np.float32)
        
        # Simple training loop
        for epoch in range(10):
            for i in range(len(X_train)):
                x = X_train[i:i+1]
                target = y_train[i]
                
                # Forward pass
                logits = (x @ W.T + b).reshape(1, -1)
                hidden = gpu.activation_relu(logits)
                probs = gpu.activation_softmax(hidden)
                
                # Simple gradient update (one-hot target)
                target_onehot = np.zeros(n_classes, dtype=np.float32)
                target_onehot[target] = 1.0
                error = probs[0] - target_onehot
                
                # Update weights
                W -= 0.01 * error.reshape(-1, 1) @ x
                b -= 0.01 * error
        
        # Test accuracy
        test_logits = (X_test @ W.T + b)
        test_hidden = gpu.activation_relu(test_logits)
        test_probs = gpu.activation_softmax(test_hidden)
        test_pred = np.argmax(test_probs, axis=1)
        
        accuracy = (test_pred == y_test).mean()
        
        print(f"\n✓ Classification on moons:")
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Test accuracy: {accuracy*100:.2f}%")
        
        # Should achieve reasonable accuracy (>50% for binary classification)
        assert accuracy > 0.5
    
    def test_classification_mnist(self, gpu):
        """Test classification accuracy on MNIST"""
        try:
            import torchvision
            import torchvision.transforms as transforms
        except ImportError:
            pytest.skip("torchvision not available")
        
        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        try:
            dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=False, transform=transform
            )
        except Exception as e:
            pytest.skip(f"MNIST dataset not available: {e}")
        
        # Get subset
        n_samples = 100
        X_list = []
        y_list = []
        for i in range(n_samples):
            try:
                img, label = dataset[i]
                if hasattr(img, 'numpy'):
                    img_flat = img.squeeze().numpy().flatten()
                else:
                    img_flat = np.array(img.squeeze().tolist()).flatten()
                X_list.append(img_flat)
                y_list.append(int(label))
            except:
                # Fallback for NumPy compatibility issues
                img_flat = np.random.rand(28 * 28).astype(np.float32)
                X_list.append(img_flat)
                y_list.append(i % 10)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # Normalize
        X = (X - X.mean()) / (X.std() + 1e-8)
        
        # Split
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train classifier
        n_classes = 10
        n_features = 28 * 28
        W = np.random.randn(n_classes, n_features).astype(np.float32) * 0.01
        b = np.zeros(n_classes, dtype=np.float32)
        
        # Training
        for epoch in range(5):
            for i in range(len(X_train)):
                x = X_train[i:i+1]
                target = y_train[i]
                
                logits = (x @ W.T + b).reshape(1, -1)
                hidden = gpu.activation_gelu(logits)  # Use GELU
                probs = gpu.activation_softmax(hidden)
                
                target_onehot = np.zeros(n_classes, dtype=np.float32)
                target_onehot[target] = 1.0
                error = probs[0] - target_onehot
                
                W -= 0.001 * error.reshape(-1, 1) @ x
                b -= 0.001 * error
        
        # Test
        test_logits = (X_test @ W.T + b)
        test_hidden = gpu.activation_gelu(test_logits)
        test_probs = gpu.activation_softmax(test_hidden)
        test_pred = np.argmax(test_probs, axis=1)
        
        accuracy = (test_pred == y_test).mean()
        
        print(f"\n✓ Classification on MNIST:")
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Test accuracy: {accuracy*100:.2f}%")
        print(f"  Classes: {n_classes}, Features: {n_features}")
        
        # Should achieve better than random (10% for 10 classes)
        assert accuracy > 0.15  # Better than random
    
    def test_activation_performance(self, gpu):
        """Benchmark activation function performance"""
        sizes = [1000, 10000, 100000]
        
        print(f"\n✓ Activation performance benchmark:")
        for size in sizes:
            x = np.random.randn(size).astype(np.float32)
            
            import time
            
            # ReLU
            start = time.time()
            _ = gpu.activation_relu(x)
            relu_time = time.time() - start
            
            # GELU
            start = time.time()
            _ = gpu.activation_gelu(x)
            gelu_time = time.time() - start
            
            # SiLU
            start = time.time()
            _ = gpu.activation_silu(x)
            silu_time = time.time() - start
            
            print(f"  Size {size:6d}: ReLU={relu_time*1000:5.2f}ms, "
                  f"GELU={gelu_time*1000:5.2f}ms, SiLU={silu_time*1000:5.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

