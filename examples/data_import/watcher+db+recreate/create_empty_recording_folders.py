from pathlib import Path

# Given
n_subjs = 50
session_names = [
    "practice",
    "studyvisit2",
    "studyvisit3",
    "studyvisit4",
    "followup1m",
    "followup3m",
    "followup6m",
]
path_dst_dir = Path.home() / "data/bids/eegacamp-test"

# Create a list of expected recordings sets
expected_recs = []
for sub_id in range(1, n_subjs + 1):
    for ses_id in session_names:
        expected_recs.append((sub_id, ses_id))

# Create recording dirs at the destination
for sub_id, ses_id in sorted(expected_recs):
    path_rec_dst = path_dst_dir / f"sub-{sub_id:03d}/ses-{ses_id}/eeg"
    path_rec_dst.mkdir(parents=True, exist_ok=True)
