import sys
sys.path.insert(0, '.')
from utils.logger import get_logger
log = get_logger('test_preprocessing')
log.info('🧹 Starting preprocessing pipeline...')
try:
    from utils.file_helper import get_latest_file
    from utils.config import RAW_DIR
    raw_path = get_latest_file(RAW_DIR)
    log.info(f'📄 Latest raw file: {raw_path}')
    print('Found raw file:', raw_path)
except Exception as e:
    log.error(f'Error: {e}')
    print('Error:', e)
