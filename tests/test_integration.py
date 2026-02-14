import pytest
from unittest.mock import MagicMock

def test_home_page_and_css_version(client):
    """Test home page loads and contains the new CSS version."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'MedFund' in response.data
    # Verify cache buster is present
    assert b'modern_style.css?v=2' in response.data

def test_user_registration_flow(client, mock_db_connection):
    """Test the full user registration flow."""
    mock_conn, mock_cursor = mock_db_connection
    
    # Setup mock to simulate no existing user (so registration proceeds)
    mock_cursor.fetchone.return_value = None 

    data = {
        'name': 'Test User',
        'mobile': '1234567890',
        'email': 'test@example.com',
        'address': '123 Test St',
        'city': 'Test City',
        'branch': 'Main',
        'acc_name': 'Test User',
        'bank_name': 'Test Bank',
        'account': '123456',
        'gpay_number': '1234567890',
        'uname': 'testuser',
        'pass': 'password123',
        'cpass': 'password123'
    }

    response = client.post('/reg_user', data=data, follow_redirects=True)
    
    assert response.status_code == 200
    # Check if we see the success message or are redirected to login
    # The view renders 'reg_user.html' with msg='success'
    assert b'Registered Successfully' in response.data

def test_user_login_flow(client, mock_db_connection):
    """Test user login flow."""
    mock_conn, mock_cursor = mock_db_connection
    
    # Mock user found in DB
    # Structure: (id, name, mobile, email, address, city, branch, acc_name, bank_name, account, gpay, uname, pass)
    mock_cursor.fetchone.return_value = (1, 'Test User', '1234567890', 'test@example.com', 'Addr', 'City', 'Br', 'Acc', 'Bank', '123', '123', 'testuser', 'password123')

    data = {
        'uname': 'testuser',
        'pass': 'password123'
    }

    response = client.post('/login', data=data, follow_redirects=True)
    
    assert response.status_code == 200
    # Should redirect to user_home or similar, or show user home content
    # Since we don't have user_home template mocked, we check for status or session
    with client.session_transaction() as sess:
        assert sess['uname'] == 'testuser'
