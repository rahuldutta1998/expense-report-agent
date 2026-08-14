import os
import shutil
import tarfile
import time
from datetime import datetime

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_SOURCE_DIR = os.path.join(BASE_DIR, "finance_memory_db")

# Destination directory in iCloud Drive
ICLOUD_BACKUP_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Expense_DB_Backups"
)
os.makedirs(ICLOUD_BACKUP_DIR, exist_ok=True)

# Retention policy (keep backups for 14 days)
RETENTION_DAYS = 14


def create_chroma_backup():
    if not os.path.exists(DB_SOURCE_DIR):
        print(f"⚠️ Source DB directory '{DB_SOURCE_DIR}' not found. Aborting backup.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_filename = f"chroma_backup_{timestamp}.tar.gz"
    backup_filepath = os.path.join(ICLOUD_BACKUP_DIR, backup_filename)

    print(f"📦 Creating archive for {DB_SOURCE_DIR}...")
    try:
        # Create a compressed tar archive of the database directory
        with tarfile.open(backup_filepath, "w:gz") as tar:
            tar.add(DB_SOURCE_DIR, arcname=os.path.basename(DB_SOURCE_DIR))

        size_mb = os.path.getsize(backup_filepath) / (1024 * 1024)
        print(f"✅ Backup created successfully: {backup_filename} ({size_mb:.2f} MB)")
        print(f"☁️ Destination: {ICLOUD_BACKUP_DIR}")

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return

    # Clean up backups older than RETENTION_DAYS
    prune_old_backups()


def prune_old_backups():
    print(f"🧹 Checking for backups older than {RETENTION_DAYS} days...")
    now = time.time()
    cutoff = now - (RETENTION_DAYS * 86400)
    deleted_count = 0

    for filename in os.listdir(ICLOUD_BACKUP_DIR):
        if filename.startswith("chroma_backup_") and filename.endswith(".tar.gz"):
            filepath = os.path.join(ICLOUD_BACKUP_DIR, filename)
            file_mtime = os.path.getmtime(filepath)

            if file_mtime < cutoff:
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