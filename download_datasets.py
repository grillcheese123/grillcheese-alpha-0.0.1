"""
Script to download datasets for testing
"""
import os
import sys

print("Downloading datasets for testing...")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

try:
    import torchvision
    import torchvision.transforms as transforms
    
    print("\n1. Downloading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    print(f"   ✓ MNIST downloaded: {len(mnist_train)} training samples")
    
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    print(f"   ✓ MNIST test set: {len(mnist_test)} test samples")
    
    print("\n2. Downloading CIFAR-10...")
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    cifar_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_cifar
    )
    print(f"   ✓ CIFAR-10 downloaded: {len(cifar_train)} training samples")
    
    cifar_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_cifar
    )
    print(f"   ✓ CIFAR-10 test set: {len(cifar_test)} test samples")
    
    print("\n✓ All datasets downloaded successfully!")
    print(f"   Data directory: {os.path.abspath('./data')}")
    
except ImportError as e:
    print(f"Error: {e}")
    print("Please install torchvision: uv add torchvision")
    sys.exit(1)
except Exception as e:
    print(f"Error downloading datasets: {e}")
    sys.exit(1)

