import secrets

# Generates a 32-character hex string (from 16 random bytes)
token = secrets.token_hex(16)
print(token)  # Example: 2befb22f1581dd4ede4821e6252d1359
