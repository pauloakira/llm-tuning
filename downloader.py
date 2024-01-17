from huggingface_hub import snapshot_download

def download_phi_2():
    repo_id = 'amgadhasan/phi-2'
    snapshot_download(repo_id=repo_id,repo_type="model", local_dir="./phi-2", local_dir_use_symlinks=False)