import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock fastapi and other dependencies before importing backend.main
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.security"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.staticfiles"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["torch"] = MagicMock()

class TestSecretEnforcement(unittest.TestCase):

    def test_main_py_fails_without_secrets(self):
        """Test that importing backend.main fails if secrets are not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.getenv", return_value=None):
                # Ensure we are testing the logic in backend.main
                if "backend.main" in sys.modules:
                    del sys.modules["backend.main"]

                with self.assertRaises(RuntimeError) as cm:
                    import backend.main
                self.assertIn("ADMIN_PASSWORD environment variable is not set", str(cm.exception))

    def test_crypto_utils_fails_without_secret_key(self):
        """Test that get_fernet fails if SECRET_KEY is not set."""
        # Ensure we are testing the logic in backend.crypto_utils
        if "backend.crypto_utils" in sys.modules:
             del sys.modules["backend.crypto_utils"]
        from backend.crypto_utils import get_fernet

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.getenv", return_value=None):
                with self.assertRaises(RuntimeError) as cm:
                    get_fernet()
                self.assertIn("SECRET_KEY environment variable is not set", str(cm.exception))

    def test_crypto_utils_succeeds_with_secret_key(self):
        """Test that get_fernet succeeds if SECRET_KEY is set."""
        if "backend.crypto_utils" in sys.modules:
             del sys.modules["backend.crypto_utils"]
        from backend.crypto_utils import get_fernet

        with patch.dict(os.environ, {"SECRET_KEY": "test-secret-key"}):
            try:
                get_fernet()
            except RuntimeError:
                self.fail("get_fernet() raised RuntimeError unexpectedly!")

if __name__ == "__main__":
    unittest.main()
