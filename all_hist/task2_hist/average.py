#!/usr/bin/env python3

import os
import re
from collections import defaultdict

import numpy as np


INPUT_DIR = "/home/gamma/colcon_ws/task2_hist"
OUTPUT_DIR = "/home/gamma/colcon_ws/task2_avg_hist"


def extract_prefix(filename: str):
	match = re.match(r"^([A-Za-z]+)", filename)
	if match is None:
		return None
	return match.group(1).lower()


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	grouped = defaultdict(list)

	for entry in sorted(os.listdir(INPUT_DIR)):
		if not entry.endswith(".npy"):
			continue

		prefix = extract_prefix(entry)
		if prefix is None:
			continue

		full_path = os.path.join(INPUT_DIR, entry)
		hist = np.load(full_path)
		grouped[prefix].append(hist)

	if not grouped:
		print("No histogram .npy files found with alphabetic prefixes.")
		return

	for prefix, hist_list in grouped.items():
		if len(hist_list) == 0:
			continue

		reference_shape = hist_list[0].shape
		if any(h.shape != reference_shape for h in hist_list):
			print(f"Skipping '{prefix}' due to inconsistent histogram shapes.")
			continue

		stacked = np.stack(hist_list, axis=0)
		average_hist = np.mean(stacked, axis=0)
		average_hist = average_hist.astype(np.float32)

		output_path = os.path.join(OUTPUT_DIR, f"{prefix}_average.npy")
		np.save(output_path, average_hist)
		print(f"Saved {output_path} from {len(hist_list)} files.")


if __name__ == "__main__":
	main()
