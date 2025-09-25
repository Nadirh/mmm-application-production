"""
Simple server-side encryption for client files.
Uses AES-256 encryption with client-specific keys.
"""
import os
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import structlog

logger = structlog.get_logger()


class FileEncryption:
    """Handles file encryption and decryption for client data."""

    def __init__(self):
        """Initialize encryption handler."""
        # Get master key from environment or generate one
        self.master_key = os.environ.get('MMM_ENCRYPTION_KEY', 'default-dev-key-change-in-production').encode()
        self.backend = default_backend()
        logger.info("Encryption handler initialized")

    def _derive_client_key(self, client_id: str) -> bytes:
        """
        Derive a client-specific encryption key from master key.

        Args:
            client_id: The client identifier

        Returns:
            32-byte key for AES-256 encryption
        """
        # Use PBKDF2 to derive a client-specific key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-256
            salt=client_id.encode() + b'mmm-salt-v1',
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.master_key)

    def encrypt_file(self, file_path: str, client_id: str) -> str:
        """
        Encrypt a file in place using client-specific key.

        Args:
            file_path: Path to the file to encrypt
            client_id: Client identifier for key derivation

        Returns:
            Path to encrypted file (same as input with .enc extension)
        """
        try:
            # Derive client key
            key = self._derive_client_key(client_id)

            # Generate random IV
            iv = os.urandom(16)  # 128 bits for AES

            # Read file content
            with open(file_path, 'rb') as f:
                plaintext = f.read()

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()

            # Pad data to multiple of 16 bytes (AES block size)
            padding_length = 16 - (len(plaintext) % 16)
            padded_plaintext = plaintext + bytes([padding_length]) * padding_length

            # Encrypt
            ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

            # Save encrypted file with IV prepended
            encrypted_path = file_path + '.enc'
            with open(encrypted_path, 'wb') as f:
                f.write(iv + ciphertext)

            # Remove original unencrypted file
            os.remove(file_path)

            logger.info(f"File encrypted for client {client_id}: {encrypted_path}")
            return encrypted_path

        except Exception as e:
            logger.error(f"Encryption failed for client {client_id}: {str(e)}")
            raise

    def decrypt_file(self, encrypted_path: str, client_id: str) -> bytes:
        """
        Decrypt a file using client-specific key.

        Args:
            encrypted_path: Path to the encrypted file
            client_id: Client identifier for key derivation

        Returns:
            Decrypted file content as bytes
        """
        try:
            # Derive client key
            key = self._derive_client_key(client_id)

            # Read encrypted file
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()

            # Extract IV (first 16 bytes) and ciphertext
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()

            # Decrypt
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            # Remove padding
            padding_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-padding_length]

            logger.info(f"File decrypted for client {client_id}")
            return plaintext

        except Exception as e:
            logger.error(f"Decryption failed for client {client_id}: {str(e)}")
            raise

    def decrypt_to_temp_file(self, encrypted_path: str, client_id: str) -> str:
        """
        Decrypt a file to a temporary location for processing.

        Args:
            encrypted_path: Path to the encrypted file
            client_id: Client identifier for key derivation

        Returns:
            Path to temporary decrypted file
        """
        import tempfile

        # Decrypt content
        plaintext = self.decrypt_file(encrypted_path, client_id)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
            temp_file.write(plaintext)
            temp_path = temp_file.name

        logger.info(f"Created temporary decrypted file: {temp_path}")
        return temp_path


# Global instance
file_encryption = FileEncryption()