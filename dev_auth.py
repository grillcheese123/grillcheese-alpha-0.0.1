"""
Developer credentials and authentication
"""
import hashlib
import os

# Developer password hash (SHA-256)
# Default password: "grillcheese_dev_2026"
# Change this by running: python -c "import hashlib; print(hashlib.sha256(b'your_password').hexdigest())"
DEV_PASSWORD_HASH = "398182f2b68931c2e2dbd9a7f65c90ed5ae682ef8399793f1afdb3dcf8fa9c74"

# Environment variable override
ENV_PASSWORD_HASH = os.environ.get("GRILLCHEESE_DEV_PASSWORD_HASH")
if ENV_PASSWORD_HASH:
    DEV_PASSWORD_HASH = ENV_PASSWORD_HASH


def verify_dev_password(password: str) -> bool:
    """
    Verify developer password
    
    Args:
        password: Password to verify
        
    Returns:
        True if password is correct
    """
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == DEV_PASSWORD_HASH


def set_dev_password(new_password: str) -> str:
    """
    Generate hash for new developer password
    
    Args:
        new_password: New password to hash
        
    Returns:
        SHA-256 hash to save in config
    """
    return hashlib.sha256(new_password.encode()).hexdigest()


# Usage instructions
USAGE = """
To set a custom developer password:
1. Generate hash: python -c "from dev_auth import set_dev_password; print(set_dev_password('your_password'))"
2. Update DEV_PASSWORD_HASH in dev_auth.py
OR
3. Set environment variable: export GRILLCHEESE_DEV_PASSWORD_HASH="your_hash"
"""
