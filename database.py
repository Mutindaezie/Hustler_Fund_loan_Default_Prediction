# database.py

# Database models and operations for user authentication and admin management

class User:
    def __init__(self, username, password, is_admin=False):
        self.username = username
        self.password = password  # In production, ensure to hash this!
        self.is_admin = is_admin

    def __repr__(self):
        return f'<User {self.username}>'

# Sample in-memory user storage
users = []

def create_user(username, password, is_admin=False):
    new_user = User(username, password, is_admin)
    users.append(new_user)
    return new_user

def authenticate(username, password):
    for user in users:
        if user.username == username and user.password == password:
            return user
    return None

# Admin operations

def list_users():
    return users

def delete_user(username):
    global users
    users = [user for user in users if user.username != username]

# Example usage (to be removed in production):
if __name__ == '__main__':
    create_user('admin', 'admin_password', True)
    create_user('user1', 'user1_password')
    print(list_users())
    print(authenticate('admin', 'admin_password'))
    delete_user('user1')
    print(list_users())
