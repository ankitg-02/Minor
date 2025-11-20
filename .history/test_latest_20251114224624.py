from utils.file_helper import get_latest_file
from utils.config import RAW_DIR
try:
    latest = get_latest_file(RAW_DIR)
    print('Latest file:', latest)
except Exception as e:
    print('Error:', e)
