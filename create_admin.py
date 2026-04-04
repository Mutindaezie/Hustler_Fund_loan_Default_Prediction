from django.contrib.auth import get_user_model

User = get_user_model()

# Create the admin user
username = 'admin'
password = 'HustlerAdmin@2024'

if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(username=username, password=password)
    print('Admin user created successfully.')
else:
    print('Admin user already exists.')