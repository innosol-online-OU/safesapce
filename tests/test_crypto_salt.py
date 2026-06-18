
import os
import sys
import unittest
from unittest.mock import patch

# Mock dependencies that might be missing in the environment
from unittest.mock import MagicMock
sys.modules['lightning-sdk'] = MagicMock()

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCryptoSalt(unittest.TestCase):
    def test_default_salt(self):
        """Test that the default salt is used when CRYPTO_SALT is not set."""
        if 'CRYPTO_SALT' in os.environ:
            del os.environ['CRYPTO_SALT']

        # Reload the module to pick up the change in SALT (since it's a module-level constant)
        import importlib
        import backend.crypto_utils as crypto_utils
        importlib.reload(crypto_utils)

        self.assertEqual(crypto_utils.SALT, b'safespace_v2_salt_2026')

    def test_dynamic_salt(self):
        """Test that a custom salt is used when CRYPTO_SALT is set."""
        os.environ['CRYPTO_SALT'] = 'custom_salt_123'

        import importlib
        import backend.crypto_utils as crypto_utils
        importlib.reload(crypto_utils)

        self.assertEqual(crypto_utils.SALT, b'custom_salt_123')

    def test_key_derivation_with_different_salts(self):
        """Test that different salts result in different Fernet keys."""
        secret = "super-secret-password"

        # Salt 1
        os.environ['CRYPTO_SALT'] = 'salt1'
        import importlib
        import backend.crypto_utils as crypto_utils
        importlib.reload(crypto_utils)
        key1 = crypto_utils.get_fernet_key(secret)

        # Salt 2
        os.environ['CRYPTO_SALT'] = 'salt2'
        importlib.reload(crypto_utils)
        key2 = crypto_utils.get_fernet_key(secret)

        self.assertNotEqual(key1, key2)

    def test_encryption_decryption_with_dynamic_salt(self):
        """Test that encryption and decryption work with a custom salt."""
        os.environ['CRYPTO_SALT'] = 'dynamic_test_salt'
        import importlib
        import backend.crypto_utils as crypto_utils
        importlib.reload(crypto_utils)

        original_data = "Sensitive information"
        secret = "another-secret"

        encrypted = crypto_utils.encrypt_data(original_data, secret)
        decrypted = crypto_utils.decrypt_data(encrypted, secret)

        self.assertEqual(original_data, decrypted)

        # Verify it fails to decrypt with a different salt
        os.environ['CRYPTO_SALT'] = 'wrong_salt'
        importlib.reload(crypto_utils)

        with self.assertRaises(Exception):
            crypto_utils.decrypt_data(encrypted, secret)

if __name__ == "__main__":
    unittest.main()
