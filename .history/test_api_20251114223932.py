from data_ingestion.api_config import get_youtube_service
try:
    service = get_youtube_service()
    print('YouTube service created successfully')
except Exception as e:
    print('Error creating YouTube service:', e)
