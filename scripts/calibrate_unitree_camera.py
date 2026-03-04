#!/usr/bin/env python3
"""Estimate camera intrinsics/distortion from checkerboard images."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate camera from checkerboard images")
    p.add_argument("--glob", required=True, help="Image glob, e.g. '/path/*.jpg'")
    p.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners (columns)")
    p.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners (rows)")
    p.add_argument("--square-size", type=float, default=0.024, help="Square size in meters")
    p.add_argument("--output", default="unitree_camera_calibration.json", help="Output JSON path")
    return p.parse_args()


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size
    return objp


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    board = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_size)
    objpoints = []
    imgpoints = []
    im_size = None

    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCornersSB(gray, board, None)
        if not found:
            found, corners = cv2.findChessboardCorners(gray, board, None)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

    if im_size is None:
        raise SystemExit("Could not read any images")
    if len(objpoints) < 8:
        raise SystemExit(f"Need at least 8 valid checkerboard images, got {len(objpoints)}")

    rms, k, d, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        im_size,
        None,
        None,
    )

    total_err = 0.0
    total_pts = 0
    for op, ip in zip(objpoints, imgpoints):
        ok, rvec, tvec = cv2.solvePnP(op, ip, k, d)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(op, rvec, tvec, k, d)
        err = cv2.norm(ip, proj, cv2.NORM_L2)
        total_err += err * err
        total_pts += len(op)
    reproj = float(np.sqrt(total_err / max(total_pts, 1)))

    out = {
        "image_width": int(im_size[0]),
        "image_height": int(im_size[1]),
        "camera_matrix": k.tolist(),
        "dist_coeffs": d.reshape(-1).tolist(),
        "rms": float(rms),
        "mean_reprojection_error_px": reproj,
        "valid_images": int(len(objpoints)),
        "board": {"cols": args.cols, "rows": args.rows, "square_size_m": args.square_size},
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    d5 = np.zeros(5, dtype=float)
    src = d.reshape(-1)
    d5[: min(5, src.size)] = src[: min(5, src.size)]

    print(f"Saved calibration: {out_path}")
    print(f"Valid images: {len(objpoints)} / {len(files)}")
    print(f"RMS: {rms:.4f} | mean reproj err: {reproj:.4f} px")
    print("")
    print("Export these on HPC before hpc_robot_bridge:")
    print(f"export HPC_ROBOT_UNDISTORT=1")
    print(f"export HPC_ROBOT_FX={fx}")
    print(f"export HPC_ROBOT_FY={fy}")
    print(f"export HPC_ROBOT_CX={cx}")
    print(f"export HPC_ROBOT_CY={cy}")
    print(f"export HPC_ROBOT_K1={d5[0]}")
    print(f"export HPC_ROBOT_K2={d5[1]}")
    print(f"export HPC_ROBOT_P1={d5[2]}")
    print(f"export HPC_ROBOT_P2={d5[3]}")
    print(f"export HPC_ROBOT_K3={d5[4]}")


if __name__ == "__main__":
    main()
