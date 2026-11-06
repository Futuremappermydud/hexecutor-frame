"""Shared naming and artifact rules for repository audits and history cleanup."""
from pathlib import PurePosixPath
import re

ARTIFACT_SUFFIXES = (
    '.bin', '.blob', '.pt', '.pth', '.ptl', '.pte', '.ckpt', '.safetensors',
    '.onnx', '.onnx.data', '.tflite', '.pb', '.fb', '.npy', '.npz', '.pkl',
    '.pickle', '.qdrx', '.nprg', '.qnn', '.raw', '.u8', '.u16', '.f16', '.f32',
    '.pool1', '.epcpu', '.ppcpu', '.ncpfx', '.mpfx', '.s2pf', '.zip', '.gz',
    '.elf', '.so', '.o', '.a', '.spv', '.pyc', '.pgm', '.asm', '.dis', '.disasm.txt',
    '.qnn.json', '.htp-strings.json', '.generated.h', '.ok',
)
ARTIFACT_DIRS = {'models', 'recovered', 'model-assets', 'quest-models', 'checkpoints',
                 'datasets', 'runs', 'build', 'dist', 'fixtures', '__pycache__',
                 '.work', 'captures', 'experiments'}
CAPTURE_HEADERS = {'space_to_depth_cases.h', 'ch5099_shiftw_partial_oracle.h'}


def is_artifact(path):
    p = PurePosixPath(path)
    parts = p.parts
    if p.name.isdigit() and p.parent.name in {'depth-model', 'quest-depth', 'qd'}:
        return True
    # Compatibility links named models/recovered are links, not artifact directories.
    if any(part in ARTIFACT_DIRS for part in parts[:-1]):
        return True
    if p.name == '.DS_Store' or p.name.endswith(ARTIFACT_SUFFIXES):
        return True
    if p.name in CAPTURE_HEADERS or p.name.endswith(('_actual_data.h', '_fixture_data.h', '_input_data.h', '_partial_oracle.h')):
        return True
    if any(parts[i:i + 2] == ('portable', 'check') for i in range(len(parts) - 1)):
        return True
    if 'metadata' in parts and any(x in parts for x in ('quest-depth', 'depth-model', 'qd')):
        # The compatibility symlink itself stays tracked; exported metadata is local.
        return parts.index('metadata') < len(parts) - 1
    return False


def embedded_model_array(data):
    """Catch captured weights/pools stored as C initializers, regardless of name."""
    if len(data) < 8192:
        return False
    for match in re.finditer(rb'\b(?:const\s+)?(?:u?int\d+_t|float|double|unsigned\s+char)\s+\w+\s*\[[^;{}]*\]\s*=\s*\{([^}]+)\}', data):
        body = match[1]
        if body.count(b',') >= 1024:
            return True
    return False


def generic_name(text):
    # Order matters: public environment names and component paths precede tokens.
    for old, new in (
        ('FRAME_QUEST_', 'FRAME_DEPTH_'), ('quest-depth', 'depth-model'),
        ('quest-models', 'model-assets'), ('quest_meta', 'model_metadata'),
        ('QUEST_META', 'MODEL_METADATA'), ('QuestMeta', 'ModelMetadata'),
        ('quest_depth', 'model_depth'), ('QuestDepth', 'ModelDepth'),
        ('Quest 3', 'Legacy headset'), ('Quest 2', 'Legacy headset'),
    ):
        text = text.replace(old, new)
    text = text.replace('Quest', 'Model')
    # Do not change request/requests/REQUEST or other unrelated words.
    text = re.sub(r'(?<![A-Za-z])quest(?=$|[^a-z]|[A-Z])', 'model', text)
    text = re.sub(r'(?<![A-Z])QUEST(?=$|[^A-Z])', 'MODEL', text)
    return text


def generic_text(data):
    if b'\0' in data:
        return data
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return data
    # Source URLs remain accurate citations even when a title uses generic wording.
    pieces = re.split(r'(https?://[^\s<>"\)\]]+)', text)
    return ''.join(part if i % 2 else generic_name(part) for i, part in enumerate(pieces)).encode('utf-8')


def main():
    import argparse
    import subprocess
    ap = argparse.ArgumentParser(description='Reject tracked model binaries and raw dumps.')
    ap.add_argument('--history', action='store_true', help='audit paths across every Git ref')
    args = ap.parse_args()
    if args.history:
        output = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], text=True)
        paths = {row.split(' ', 1)[1] for row in output.splitlines() if ' ' in row}
        objects = {row.split(' ', 1)[0]: row.split(' ', 1)[1] if ' ' in row else row
                   for row in output.splitlines()}
    else:
        output = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).decode()
        entries = [row.split('\t', 1) for row in output.strip('\0').split('\0') if row]
        paths = {path for _metadata, path in entries}
        objects = {metadata.split()[1]: path for metadata, path in entries}
    failures = sorted(path for path in paths if is_artifact(path))
    if failures:
        print('\n'.join(failures))
        raise SystemExit(f'{len(failures)} artifact paths found')
    process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    checked = 0
    try:
        for oid, path in objects.items():
            process.stdin.write((oid + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError(f'Cannot audit Git object {oid}')
            data = process.stdout.read(int(header[2]))
            assert process.stdout.read(1) == b'\n'
            if header[1] != b'blob':
                continue
            checked += 1
            image = data.startswith((b'\x89PNG', b'\xff\xd8\xff', b'\x00\x00\x01\x00'))
            try:
                data.decode('utf-8')
                binary = b'\0' in data
            except UnicodeDecodeError:
                binary = True
            if embedded_model_array(data) or (binary and not image):
                failures.append(path)
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if failures:
        print('\n'.join(sorted(set(failures))))
        raise SystemExit('Binary dumps or embedded arrays found in tracked Git blobs')
    print(f'PASS: {len(paths)} paths and {checked} blobs audited; no model binaries or raw dumps tracked.')


if __name__ == '__main__':
    main()
