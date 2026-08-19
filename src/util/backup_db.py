import os
import shutil
import tarfile
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load local environment if .env exists
load_dotenv()

# --- Dynamic Path Resolution ---
# 1. Project Root: Resolves dynamically based on this file's position (src/util/backup_db.py -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_SOURCE_DIR = BASE_DIR / "finance_memory_db"

# 2. User Home Directory: Dynamic lookup with robust fallback
USER_HOME = Path.home() if str(Path.home()) not in ("", ".") else Path(os.path.expanduser("~"))

# 3. iCloud Backup Directory: Reads from .env if defined, else defaults to standard macOS iCloud path
DEFAULT_ICLOUD_DIR = USER_HOME / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Expense_DB_Backups"
ICLOUD_BACKUP_DIR = Path(os.getenv("ICLOUD_BACKUP_DIR", DEFAULT_ICLOUD_DIR))

# Ensure destination exists
os.makedirs(str(ICLOUD_BACKUP_DIR), exist_ok=True)

RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "14"))


def create_chroma_backup():
    if not DB_SOURCE_DIR.exists():
        print(f"⚠️ Source DB directory '{DB_SOURCE_DIR.name}' not found. Aborting backup.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_filename = f"chroma_backup_{timestamp}.tar.gz"
    backup_filepath = ICLOUD_BACKUP_DIR / backup_filename

    print(f"📦 Creating archive for {DB_SOURCE_DIR.name}...")
    try:
        with tarfile.open(backup_filepath, "w:gz") as tar:
            tar.add(str(DB_SOURCE_DIR), arcname=DB_SOURCE_DIR.name)

        size_mb = os.path.getsize(backup_filepath) / (1024 * 1024)
        print(f"✅ Backup created: {backup_filename} ({size_mb:.2f} MB)")
        print(f"☁️ Destination: {ICLOUD_BACKUP_DIR}")

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return

    prune_old_backups()


def prune_old_backups():
    print(f"🧹 Checking for backups older than {RETENTION_DAYS} days...")
    now = time.time()
    cutoff = now - (RETENTION_DAYS * 86400)
    deleted_count = 0

    for filename in os.listdir(ICLOUD_BACKUP_DIR):
        if filename.startswith("chroma_backup_") and filename.endswith(".tar.gz"):
            filepath = ICLOUD_BACKUP_DIR / filename
            if os.path.getmtime(filepath) < cutoff:
                try:
                    os.remove(filepath)
                    print(f"🗑️ Pruned old backup: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")

    if deleted_count == 0:
        print("✨ No old backups needed pruning.")


if __name__ == "__main__":
    create_chroma_backup()
