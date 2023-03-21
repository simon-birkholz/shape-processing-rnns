import os

if os.name == 'nt':
    os.add_dll_directory(r'C:\Program Files\pthread\dll\x64')
    os.add_dll_directory(r'C:\Program Files\libjpeg-turbo64\bin')
    os.add_dll_directory(r'C:\Program Files\opencv\build\x64\vc16\bin')