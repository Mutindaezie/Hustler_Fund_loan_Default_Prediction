import os
import django

# 🔧 Initialize Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Hustler_Fund_loan_Default_Prediction.settings')
django.setup()

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
