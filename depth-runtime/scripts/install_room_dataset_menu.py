#!/usr/bin/env python3
"""Install Room Dataset Recorder in Steam's non-Steam library."""
import os
import struct
import time
import zlib
from pathlib import Path

HOME = Path.home()
ROOT = HOME / 'frame-spatial-runtime'
LAUNCH = ROOT / 'scripts/launch_room_dataset.sh'
NAME = 'Room Dataset Recorder'


def cstr(data, pos):
    end = data.index(0, pos)
    return data[pos:end].decode('utf-8'), end + 1


def skip_object(data, pos):
    while data[pos] != 8:
        kind = data[pos]
        _, pos = cstr(data, pos + 1)
        if kind == 0:
            pos = skip_object(data, pos)
        elif kind == 1:
            _, pos = cstr(data, pos)
        elif kind in (2, 6):
            pos += 4
        elif kind == 7:
            pos += 8
        else:
            raise ValueError(f'unsupported Steam VDF type {kind}')
    return pos + 1


def string(key, value):
    return b'\x01' + key.encode() + b'\0' + value.encode() + b'\0'


def integer(key, value):
    return b'\x02' + key.encode() + b'\0' + struct.pack('<I', value)


def main():
    if not LAUNCH.is_file():
        raise FileNotFoundError(LAUNCH)
    configs = sorted((HOME / '.local/share/Steam/userdata').glob('*/config/shortcuts.vdf'))
    if not configs:
        raise FileNotFoundError('Steam userdata shortcuts.vdf not found')
    for path in configs:
        data = path.read_bytes()
        if not data.startswith(b'\x00shortcuts\x00'):
            raise ValueError(f'unexpected Steam VDF header: {path}')
        pos = len(b'\x00shortcuts\x00')
        keys = []
        while data[pos] != 8:
            if data[pos] != 0:
                raise ValueError(f'unexpected Steam shortcut entry at {pos}')
            key, pos = cstr(data, pos + 1)
            keys.append(key)
            pos = skip_object(data, pos)
        if NAME.encode() + b'\0' in data:
            print(f'Steam: already installed in {path}')
            continue
        index = max((int(k) for k in keys if k.isdecimal()), default=-1) + 1
        quoted = '"' + str(LAUNCH) + '"'
        appid = zlib.crc32((quoted + NAME).encode()) | 0x80000000
        entry = (b'\x00' + str(index).encode() + b'\0' + integer('appid', appid) +
                 string('AppName', NAME) + string('Exe', quoted) +
                 string('StartDir', '"' + str(ROOT) + '"') +
                 string('icon', 'camera-photo') + string('ShortcutPath', '') +
                 string('LaunchOptions', '') + integer('IsHidden', 0) +
                 integer('AllowDesktopConfig', 1) + integer('AllowOverlay', 1) +
                 integer('OpenVR', 1) + integer('Devkit', 0) +
                 string('DevkitGameID', '') + integer('DevkitOverrideAppID', 0) +
                 integer('LastPlayTime', 0) + string('FlatpakAppID', '') +
                 string('sortas', '') + b'\x00tags\x00\x08\x08')
        backup = path.with_name(f'{path.name}.backup-room-dataset-{int(time.time())}')
        backup.write_bytes(data)
        updated = data[:pos] + entry + data[pos:]
        temp = path.with_name(path.name + '.room-dataset.tmp')
        temp.write_bytes(updated)
        os.replace(temp, path)
        print(f'Steam: added {NAME} to {path}; backup: {backup}')


if __name__ == '__main__':
    main()
