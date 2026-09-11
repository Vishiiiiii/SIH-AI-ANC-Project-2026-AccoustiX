import pathlib
import os

def detect_and_convert(filepath):
    """Detect encoding and convert to UTF-8."""
    raw = filepath.read_bytes()
    
    # Check for BOM
    if raw[:2] == b'\xff\xfe':
        # UTF-16 LE
        content = raw.decode('utf-16-le')
        encoding = 'UTF-16 LE'
    elif raw[:2] == b'\xfe\xff':
        # UTF-16 BE
        content = raw.decode('utf-16-be')
        encoding = 'UTF-16 BE'
    elif raw[:3] == b'\xef\xbb\xbf':
        # UTF-8 with BOM
        content = raw.decode('utf-8-sig')
        encoding = 'UTF-8 BOM'
    else:
        # Assume UTF-8
        try:
            content = raw.decode('utf-8')
            encoding = 'UTF-8'
        except:
            print(f'  Cannot decode: {filepath}')
            return False
    
    # Remove null bytes that may exist in content
    content = content.replace('\x00', '')
    
    # Write as clean UTF-8
    filepath.write_text(content, encoding='utf-8')
    
    if encoding != 'UTF-8':
        print(f'  FIXED ({encoding} → UTF-8): {filepath.name}')
    return True

# Process all .py files
src_path = pathlib.Path('src')
count = 0
for py_file in src_path.rglob('*.py'):
    try:
        detect_and_convert(py_file)
        count += 1
    except Exception as e:
        print(f'  ERROR: {py_file} - {e}')

print(f'\nProcessed {count} files')