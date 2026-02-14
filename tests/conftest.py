import pytest
import sys
from unittest.mock import MagicMock

# Mock libraries that might not be installed or require heavy dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['imagehash'] = MagicMock()
sys.modules['fitz'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['diff_match_patch'] = MagicMock()
sys.modules['spire.doc'] = MagicMock()
sys.modules['spire.doc.common'] = MagicMock()
sys.modules['pytesseract'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.metrics'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock MySQL connector before importing main
mock_mysql = MagicMock()
sys.modules['mysql'] = MagicMock()
sys.modules['mysql.connector'] = mock_mysql

# Add parent dir to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.secret_key = 'test_secret'
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_db_connection(monkeypatch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Mocking get_db_connection in main module
    def mock_get_db():
        return mock_conn
        
    monkeypatch.setattr('main.get_db_connection', mock_get_db)
    
    return mock_conn, mock_cursor
