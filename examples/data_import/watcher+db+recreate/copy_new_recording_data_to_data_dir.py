import re
from pathlib import Path

# Given
path_src_dir = Path.home() / "data/bids/eegacamp"
path_dst_dir = Path.home() / "data/bids/eegacamp-test"

# Create list of recordings at source
recs_src = []
for root, dirs, files in path_src_dir.walk():
    if "derivatives" in dirs:
        dirs.remove("derivatives")
    if re.match(r"^.+/EA-\d{3}/ses-.+/eeg$", str(root)):
        sub_id = int(re.findall(r"^.+/EA-(\d{3})/ses-.+/eeg", str(root))[0])
        ses_id = re.findall(r"^.+/EA-\d{3}/ses-(.+)/eeg", str(root))[0]
        recs_src.append((sub_id, ses_id))
# print(sorted(recs_src))

# Create recording paths at the destination
for sub_id, ses_id in sorted(recs_src):
    path_rec_src = path_src_dir / f"EA-{sub_id:03d}/ses-{ses_id}/eeg"
    path_rec_dst = path_dst_dir / f"sub-{sub_id:03d}/ses-{ses_id}/eeg"

    #   Filter: copy only recordings from source which are unavailable at destination
    if (
        len([item for item in path_rec_dst.iterdir() if not item.name.startswith(".")])
        > 0
    ):
        print(f"Recording already exists: {path_rec_dst}. Skipping...")
    else:
        print(f"Copying recording: {path_rec_dst}")
        path_rec_dst.mkdir(parents=True, exist_ok=True)
        for item in path_rec_src.iterdir():
            item.copy(path_rec_dst / item.name.replace("EA-", "sub-", 1))

        #   Create an empty file marking completion of data copying
        path_data_completion_flag = path_rec_dst / ".data_collection_completed"
        path_data_completion_flag.touch()
