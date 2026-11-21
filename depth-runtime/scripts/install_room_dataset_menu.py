#!/usr/bin/env python3
"""Install Room Dataset Recorder in Steam's non-Steam library."""
import os
import struct
import time
import zlib
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'scripts/launch_room_dataset.sh'
NAME = 'Room Dataset Recorder'


def cstr(data, pos):
    end = data.index(0, pos)
    return data[pos:end].decode('utf-8'), end + 1


def skip_object(data, pos, strings=None):
    while data[pos] != 8:
        kind = data[pos]
        key, pos = cstr(data, pos + 1)
        if kind == 0:
            pos = skip_object(data, pos)
        elif kind == 1:
            start = pos
            value, pos = cstr(data, pos)
            if strings is not None:
                strings[key] = (value, start, pos)
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
        matches = []
        while data[pos] != 8:
            if data[pos] != 0:
                raise ValueError(f'unexpected Steam shortcut entry at {pos}')
            key, pos = cstr(data, pos + 1)
            keys.append(key)
            fields = {}
            pos = skip_object(data, pos, fields)
            if fields.get('AppName', (None,))[0] == NAME:
                matches.append(fields)
        quoted = '"' + str(LAUNCH) + '"'
        start_dir = '"' + str(ROOT) + '"'
        if matches:
            edits = []
            for fields in matches:
                for key, value in (('Exe', quoted), ('StartDir', start_dir)):
                    previous, start, end = fields[key]
                    if previous != value:
                        edits.append((start, end, value.encode() + b'\0'))
            updated = data
            for start, end, value in sorted(edits, reverse=True):
                updated = updated[:start] + value + updated[end:]
            if updated == data:
                print(f'Steam: already installed in {path}')
                continue
            action = 'updated'
        else:
            index = max((int(k) for k in keys if k.isdecimal()), default=-1) + 1
            appid = zlib.crc32((quoted + NAME).encode()) | 0x80000000
            entry = (b'\x00' + str(index).encode() + b'\0' + integer('appid', appid) +
                     string('AppName', NAME) + string('Exe', quoted) +
                     string('StartDir', start_dir) +
                     string('icon', 'camera-photo') + string('ShortcutPath', '') +
                     string('LaunchOptions', '') + integer('IsHidden', 0) +
                     integer('AllowDesktopConfig', 1) + integer('AllowOverlay', 1) +
                     integer('OpenVR', 1) + integer('Devkit', 0) +
                     string('DevkitGameID', '') + integer('DevkitOverrideAppID', 0) +
                     integer('LastPlayTime', 0) + string('FlatpakAppID', '') +
                     string('sortas', '') + b'\x00tags\x00\x08\x08')
            updated = data[:pos] + entry + data[pos:]
            action = 'added'
        backup = path.with_name(f'{path.name}.backup-room-dataset-{int(time.time())}')
        backup.write_bytes(data)
        temp = path.with_name(path.name + '.room-dataset.tmp')
        temp.write_bytes(updated)
        os.replace(temp, path)
        print(f'Steam: {action} {NAME} to {path}; backup: {backup}')


if __name__ == '__main__':
    main()
