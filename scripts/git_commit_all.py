import dulwich.porcelain as porcelain
from pathlib import Path

repo_path = Path(__file__).resolve().parents[1]

# Stage all files
print(f"Staging all files in {repo_path}...")
porcelain.add(str(repo_path), paths=["."])

# Commit changes
commit_id = porcelain.commit(
    str(repo_path),
    message=b"Initial commit: AtlasML full pipeline (steps 1-14)",
    author=b"User <user@atlasml.local>",
)

print(f"Committed successfully! Commit ID: {commit_id.decode('utf-8')}")
