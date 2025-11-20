import os
print('File size:', os.path.getsize('logs/project_20251114.log'))
print('Is empty:', os.path.getsize('logs/project_20251114.log') == 0)
print('Trying to read:')
try:
    with open('logs/project_20251114.log', 'r', encoding='utf-8') as f:
        content = f.read()
        print('Content:', repr(content))
        print('Length:', len(content))
except Exception as e:
    print('Error:', e)
