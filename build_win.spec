# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis([SCRIPT_PATH],  # replace SCRIPT_PATH with absolute path to enlarger_gooey.py in this direcotry
             pathex=[SCRIPT_PATH], # replace SCRIPT_PATH with absolute path to enlarger_gooey.py in this direcotry
             binaries=[],
             datas=[(ONNXRUNTIME_DLL_PATH, r".\onnxruntime\capi"), Replace ONNXRUNTIME_DLL_PATH with absolute path to onnxruntime_providers_shared.dll (Check YOUR-PYTHON-ENV\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll)
                    (IMAGES_PATH, './images'), Replace IMAGES_PATH with absolute path to images folder in this direcotry
                    ("./models/model-4x.onnx", './models')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='4xImageEnlarger-v0.1',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon=os.path.join(r'images\program_icon.ico'))