import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'\bNEXUS\b', 'SENTINEL AI', content)
content = re.sub(r'\bnexus\b', 'sentinel_ai', content)
content = re.sub(r'\bNexus\b', 'Sentinel AI', content)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Renamed successfully!')
