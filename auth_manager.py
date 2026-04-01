import bcrypt

class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hashes a password using bcrypt."""
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed.decode('utf-8')

    @staticmethod
    def validate_password(password: str, hashed: str) -> bool:
        """Validates a password against a hashed password."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def authenticate(username: str, password: str, user_data: dict) -> bool:
        """Authenticates a user by username and password."""
        if username in user_data:
            return AuthManager.validate_password(password, user_data[username]['password'])
        return False

# Example usage:
# user_data = {'user1': {'password': AuthManager.hash_password('mysecurepassword')}}
# print(AuthManager.authenticate('user1', 'mysecurepassword', user_data))