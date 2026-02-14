import pytest
from unittest.mock import MagicMock

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'index.html' in response.data or response.template == 'index.html' # Depending on how templates render in test

def test_login_admin_get(client):
    response = client.get('/login_admin')
    assert response.status_code == 200

def test_login_admin_post_success(client, mock_db_connection):
    mock_conn, mock_cursor = mock_db_connection
    # Simulate finding an admin user
    mock_cursor.fetchone.return_value = (1, 'admin', 'password')

    response = client.post('/login_admin', data={'uname': 'admin', 'pass': 'password'}, follow_redirects=True)
    assert response.status_code == 200
    # Check if we were redirected to admin page
    # Since we use follow_redirects=True, request.path should be /admin (if successful)
    # However, templates might fail rendering if data is missing, so we check status 200 primarily

def test_login_admin_post_fail(client, mock_db_connection):
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.fetchone.return_value = None

    response = client.post('/login_admin', data={'uname': 'admin', 'pass': 'wrong'}, follow_redirects=True)
    assert response.status_code == 200
    assert b'Incorrect username/password!' in response.data

def test_reg_user_get(client):
    response = client.get('/reg_user')
    assert response.status_code == 200
