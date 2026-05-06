from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def model_slug(model_id: str | None) -> str:
    import re

    raw = str(model_id or 'default').strip().lower().replace('\\', '/')
    raw = re.sub(r'[^a-z0-9._/-]+', '-', raw)
    raw = raw.replace('/', '-').replace('_', '-')
    raw = re.sub(r'-+', '-', raw).strip('-.')
    return raw or 'default'


def read_base_model(adapter_dir: Path) -> str | None:
    cfg = adapter_dir / 'adapter_config.json'
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding='utf-8'))
        return str(data.get('base_model_name_or_path') or data.get('base_model_name') or '').strip() or None
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: python scripts/migrate_adapter_layout.py <adapter_latest_dir>')
        return 2

    latest_dir = Path(sys.argv[1]).expanduser()
    if latest_dir.name != 'latest':
        print(f'Expected a path ending in latest, got: {latest_dir}')
        return 2

    if not latest_dir.exists():
        print(f'No adapter found at {latest_dir}; nothing to migrate.')
        return 0

    base_model = read_base_model(latest_dir)
    if not base_model:
        print(f'Could not determine base model from {latest_dir / "adapter_config.json"}; leaving in place.')
        return 1

    dest = latest_dir.parent / model_slug(base_model) / 'latest'
    if dest.resolve() == latest_dir.resolve():
        print(f'Adapter already in model-specific location: {dest}')
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f'Destination already exists: {dest}')
        return 1

    shutil.move(str(latest_dir), str(dest))
    print(f'Moved adapter for {base_model} -> {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
