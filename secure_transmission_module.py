from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets

def encrypt(data, key):
    # Generate a random initialization vector (IV)
    iv = secrets.token_bytes(16)  # IV size should match the block size of the encryption algorithm

    # Create the Cipher object with the specified IV and feedback size (8 bits)
    cipher = Cipher(algorithms.AES(key), modes.CFB8(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Encrypt the data
    encrypted_data = encryptor.update(data) + encryptor.finalize()

    # Return the IV along with the encrypted data
    return iv + encrypted_data

def decrypt(encrypted_data, key):
    # Extract the IV from the first 16 bytes of the encrypted data
    iv = encrypted_data[:16]
    encrypted_data = encrypted_data[16:]

    # Create the Cipher object with the specified IV and feedback size (8 bits)
    cipher = Cipher(algorithms.AES(key), modes.CFB8(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the data
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    return decrypted_data