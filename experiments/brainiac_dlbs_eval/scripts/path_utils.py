from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[3]


def resolve_from_repo(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()
