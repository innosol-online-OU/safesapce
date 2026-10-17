"""
Crypto Utilities for SafeSpace History Encryption.
Uses Fernet symmetric encryption with key derived from SECRET_KEY.
"""
import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Salt for key derivation (fetched from env for security)
SALT = os.getenv('CRYPTO_SALT', 'safespace_v2_salt_2026').encode()

def get_fernet_key(secret_key: str) -> bytes:
    """
    Derive a Fernet-compatible key from a password/secret.
    Uses PBKDF2 with SHA256.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
    return key

def get_fernet(secret_key: str = None) -> Fernet:
    """Get a Fernet instance using SECRET_KEY from env or parameter."""
    if secret_key is None:
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key:
            raise RuntimeError("SECRET_KEY environment variable is not set")
    key = get_fernet_key(secret_key)
    return Fernet(key)

def encrypt_data(data: str, secret_key: str = None) -> str:
    """Encrypt a string and return base64-encoded ciphertext."""
    f = get_fernet(secret_key)
    encrypted = f.encrypt(data.encode())
    return encrypted.decode()

def decrypt_data(encrypted: str, secret_key: str = None) -> str:
    """Decrypt base64-encoded ciphertext and return plaintext."""
    f = get_fernet(secret_key)
    decrypted = f.decrypt(encrypted.encode())
    return decrypted.decode()
