import math
import os
import subprocess


def chunked_commit(directory, chunk_size=100, commit_message="Chunked commit"):
    os.chdir(directory)

    # Get list of all files in the directory recursively
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    new_files = result.stdout.splitlines()

    if not new_files:
        print("No new files to commit.")
        return

    total_files = len(new_files)
    total_batches = math.ceil(total_files / chunk_size)
    print(f"Found {total_files} files to commit in {total_batches} batches.")

    # Commit files in chunks
    for i in range(0, len(new_files), chunk_size):
        current_batch = i // chunk_size + 1
        chunk = new_files[i : i + chunk_size]

        # Stage files
        subprocess.run(["git", "add"] + chunk)

        # Commit files
        subprocess.run(
            ["git", "commit", "-m", f"{commit_message} (Batch {i // chunk_size + 1})"]
        )

        print(
            f"Committed batch {current_batch}/{total_batches} with {len(chunk)} files."
        )

        print(f"Pushing batch {current_batch}/{total_batches}...")
        subprocess.run(["git", "push"])
        print(f"Batch {current_batch}/{total_batches} pushed!")

    print("All files committed in chunks.")


# Example usage
if __name__ == "__main__":
    folder_path = "phase_acts/val"  # Change this to your target folder
    chunked_commit(folder_path, chunk_size=100, commit_message="Auto chunk commit")
