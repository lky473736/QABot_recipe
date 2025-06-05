import zipfile
import os

def zip_current_directory(output_filename='archive.zip'):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            for file in files:
                filepath = os.path.join(root, file)
                # zip 파일에 저장할 때는 상대경로로 추가
                arcname = os.path.relpath(filepath, '.')
                zipf.write(filepath, arcname)

zip_current_directory('my_archive.zip')
