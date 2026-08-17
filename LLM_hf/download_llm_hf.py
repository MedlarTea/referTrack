from huggingface_hub import snapshot_download


def main() -> None:
    # This release uses Qwen3-4B as the LLM backbone.
    repo_ids_and_local_dirs = {
        "Qwen/Qwen3-4B": "./qwen3-4b",
    }
    for repo_id, local_dir in repo_ids_and_local_dirs.items():
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )


if __name__ == "__main__":
    main()
