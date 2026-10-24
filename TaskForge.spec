# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ClearCommit TaskForge (onedir).
# engine/ is on pathex so the modules are ANALYZED as source (their stdlib
# imports like wave/... get bundled). ui/ ships as data for the webview.

a = Analysis(
    ['app.py'],
    pathex=[r'D:\ClearCommit-TaskForge\engine'],
    binaries=[],
    datas=[('ui', 'ui')],
    hiddenimports=[
        'stream_pipeline', 'transcriber', 'audio_capture', 'loopback_capture',
        'config', 'logger', 'utils', 'translator', 'nvidia_stream',
        'nvidia_translator', 'deepgram_tts', 'elevenlabs_ws',
        'ai_service', 'ai_providers', 'webview',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "psutil", "IPython", "jedi", "pandas", "PyQt5", "PySide2", "PyQt6",
        "PySide6", "torch", "tensorflow", "scipy", "matplotlib",
        "gradio", "discord", "numpy.tests", "pytest", "tkinter.test", "faster_whisper", "ctranslate2",
        "transformers", "pydub", "soundfile", "av",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ClearCommitTaskForge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='ClearCommitTaskForge',
)
