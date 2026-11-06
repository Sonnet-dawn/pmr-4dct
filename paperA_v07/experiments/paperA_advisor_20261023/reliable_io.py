"""Bounded retries for Windows readers temporarily blocking atomic replacement."""
import os
import tempfile
import time
from pathlib import Path


def atomic_text(path, text, *, required=True, attempts=40):
    path = Path(path)
    handle, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
        for trial in range(attempts):
            try:
                os.replace(str(temporary), str(path))
                return True
            except PermissionError:
                if trial+1 == attempts:
                    if required:
                        raise
                    return False
                time.sleep(min(.02*(trial+1), .15))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass

