# Use cleaned agent by default, fallback to original if needed

import argparse
import os
from pathlib import Path
import json

# GT↔catalog IoU thresh used by ReferAgent (REFER_TRACKER_CFG["target_iou_thresh"])
GT_IOU_THRESH = 0.2


def _is_invisible(gt_bbox_px) -> bool:
    """Target not visible: sensor writes [0,0,0,0] (None treated the same)."""
    if gt_bbox_px is None:
        return True
    return all(float(x) == 0.0 for x in gt_bbox_px[:4])


def _is_included(f) -> bool:
    """Target is in the catalog: IoU with any candidate >= GT_IOU_THRESH."""
    gt_iou = f.get("gt_iou")
    return gt_iou is not None and float(gt_iou) >= GT_IOU_THRESH


def _noexist_correct(f) -> bool:
    """Correct NO_EXIST on an invisible frame. Prefer cot_correct, else cot_noexist."""
    if f.get("cot_correct") is True:
        return True
    if f.get("cot_correct") is False:
        return False
    return f.get("cot_noexist") is True


def _accumulate_refer_id_metrics(seq_info, counters: dict) -> None:
    """Five refer metrics:
    1) Det/CIR    = included / visible
    2) RecogAcc   = included_ok / included
    3) NoExistAcc = noexist_ok / invisible
    4) IDSwitch   = tid changes on adjacent visible frames with valid tids
    5) ReID       = first visible frame after an occlusion gap is cot_correct
    """
    if not seq_info or "gt_bbox_px" not in seq_info[0]:
        return

    for f in seq_info:
        counters["frames"] += 1
        if _is_invisible(f.get("gt_bbox_px")):
            counters["invisible"] += 1
            if _noexist_correct(f):
                counters["noexist_ok"] += 1
            continue

        counters["visible"] += 1
        if not _is_included(f):
            continue
        counters["included"] += 1
        if f.get("cot_correct") is True:
            counters["included_ok"] += 1

    # (4) ID switch: visible target and both frames have a valid track id.
    for a, b in zip(seq_info, seq_info[1:]):
        if _is_invisible(a.get("gt_bbox_px")) or _is_invisible(b.get("gt_bbox_px")):
            continue
        ta, tb = a.get("target_tid"), b.get("target_tid")
        if ta is None or tb is None:
            continue
        ta, tb = int(ta), int(tb)
        if ta < 0 or tb < 0:
            continue
        counters["id_pairs"] += 1
        if ta != tb:
            counters["id_switches"] += 1

    # (5) ReID after occlusion: invisible gap with visible frames on both sides.
    #     Success = first visible frame after the gap is cot_correct.
    T = len(seq_info)
    i = 0
    while i < T:
        if not _is_invisible(seq_info[i].get("gt_bbox_px")):
            i += 1
            continue
        gs = i
        while i < T and _is_invisible(seq_info[i].get("gt_bbox_px")):
            i += 1
        ge = i - 1  # inclusive
        if gs == 0 or ge == T - 1:
            continue  # need pre/post visible frames
        counters["reid_gaps"] += 1
        if seq_info[ge + 1].get("cot_correct") is True:
            counters["reid_ok"] += 1

def print_evaluation_results(save_path: str, split: str):
    base_result_dir = os.path.join(save_path, split)
    sub_dirs = os.listdir(base_result_dir)

    total_num = 0
    success_num = 0
    failed_num = 0
    following_rate = 0
    collision_num = 0
    far_in_failed_num = 0
    close_in_failed_num = 0
    unnormal_num = 0
    refer = {
        "frames": 0,
        "visible": 0,
        "included": 0,
        "included_ok": 0,
        "invisible": 0,
        "noexist_ok": 0,
        "id_pairs": 0,
        "id_switches": 0,
        "reid_gaps": 0,
        "reid_ok": 0,
    }
    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_result_dir, sub_dir)
        dir_path = Path(dir_path)
        collision_ids = []
        far_in_failed_ids = []
        close_in_failed_ids = []
        failed_ids = []

        for p in dir_path.glob("*.json"):
            if not p.stem.isdigit():
                continue
            
            with open(p, "r") as f:
                data = json.load(f)
            
            # open sequence info
            with open(os.path.join(dir_path, "{}_info.json".format(p.stem)), "r") as f:
                seq_info = json.load(f)
            last_item = seq_info[-1]
            dist_hr = last_item["dis_to_human"]
            facing = last_item["facing"]

            success_num += data["success"]
            collision_num += data["collision"]
            following_rate += data["following_rate"]
            _accumulate_refer_id_metrics(seq_info, refer)

            total_num += 1

            if data["collision"]:
                collision_ids.append(p.stem)
            
            if data["success"] == 0:
                failed_ids.append(p.stem)
                if data["collision"] == 0 and dist_hr > 2.0:
                    far_in_failed_ids.append(p.stem)
                if data["collision"] == 0 and dist_hr <= 1.0:
                    close_in_failed_ids.append(p.stem)
        
        # collision_all_ids.extend(collision_ids)
        # if len(collision_ids) > 0:
            # print("Collision in {}: {}".format(sub_dir, sorted(collision_ids)))
        if len(failed_ids) > 0:
            # print("Failed in {}: {}".format(sub_dir, failed_ids))
            failed_num += len(failed_ids)
            unnormal_ids = []
            for fid in failed_ids:
                if fid not in collision_ids and fid not in far_in_failed_ids and fid not in close_in_failed_ids:
                    unnormal_ids.append(fid)
            if len(unnormal_ids) > 0:
                # print("Unnormal Failed in {}: {}".format(sub_dir, unnormal_ids))
                unnormal_num += len(unnormal_ids)
        if len(far_in_failed_ids) > 0:
            # print("Far-in Failed in {}: {}".format(sub_dir, far_in_failed_ids))
            far_in_failed_num += len(far_in_failed_ids)
        if len(close_in_failed_ids) > 0:
            # print("Close-in Failed in {}: {}".format(sub_dir, close_in_failed_ids))
            close_in_failed_num += len(close_in_failed_ids)
    
    print("Failed/Total: {}/{}".format(failed_num, total_num))
    # print("Collision/Non-Collision: {}/{}".format(int(collision_num), int(failed_num-collision_num)))
    print("Collision/Far/Close/Unnormal in Failed: {}/{}/{}/{}".format(int(collision_num), far_in_failed_num, close_in_failed_num, unnormal_num))
    # print("Unnormal Failed Num: {}".format(unnormal_num))
    print("SR: {:.1f}%, TR: {:.1f}%, CR: {:.1f}%".format((success_num/total_num)*100, (following_rate/total_num)*100, (collision_num/total_num)*100))
    if unnormal_num > 0:
        adj_success = success_num + unnormal_num
        # print("SR(+unnormal): {:.1f}%, TR: {:.1f}%, CR: {:.1f}%".format((adj_success/total_num)*100, (following_rate/total_num)*100, (collision_num/total_num)*100))
    # print("Collision Num: {}".format(collision_num))

    if refer["frames"] > 0:
        det = (refer["included"] / refer["visible"] * 100) if refer["visible"] else 0.0
        recog = (refer["included_ok"] / refer["included"] * 100) if refer["included"] else 0.0
        noex = (refer["noexist_ok"] / refer["invisible"] * 100) if refer["invisible"] else 0.0
        sw_rate = (refer["id_switches"] / refer["id_pairs"] * 100) if refer["id_pairs"] else 0.0
        reid_rate = (refer["reid_ok"] / refer["reid_gaps"] * 100) if refer["reid_gaps"] else 0.0
        print(
            "Det: {:.1f}% ({}/{}), Recog: {:.1f}% ({}/{}), NoExist: {:.1f}% ({}/{}), "
            "IDSwitch: {} ({:.1f}% of {}), ReID: {} ({:.1f}% of {})".format(
                det, refer["included"], refer["visible"],
                recog, refer["included_ok"], refer["included"],
                noex, refer["noexist_ok"], refer["invisible"],
                refer["id_switches"], sw_rate, refer["id_pairs"],
                refer["reid_ok"], reid_rate, refer["reid_gaps"],
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Habitat ReferTrack eval jsons")
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory that contains split subfolders, e.g. "
             "data/logs/ReferTrack-Qwen3-4B/eval_sim_refer_refertrack_qwen3_4b",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dt"],
        help="Split folder names under --result-dir (default: dt)",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.result_dir):
        raise SystemExit(f"result dir not found: {args.result_dir}")
    print(f"==============={args.result_dir}===============")
    for split in args.splits:
        split_path = os.path.join(args.result_dir, split)
        if not os.path.isdir(split_path):
            print(f"-------{split}------- (missing, skip)")
            continue
        print(f"-------{split}-------")
        print_evaluation_results(args.result_dir, split)
