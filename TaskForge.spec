# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ClearCommit TaskForge (onedir).
# The engine/ modules are imported at runtime from a sys.path insert, so they
# ship as data files next to app.py; ui/ ships as data for the webview.

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('ui', 'ui'), ('engine', 'engine')],
    hiddenimports=[
        'stream_pipeline', 'transcriber', 'audio_capture', 'loopback_capture',
        'config', 'logger', 'utils', 'translator', 'nvidia_stream',
        'nvidia_translator', 'deepgram_tts', 'elevenlabs_ws',
        'ai_service', 'ai_providers', 'webview',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL', 'pytest'],
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
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='ClearCommitTaskForge',
)
