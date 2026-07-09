Minecraft Ai for the lonely souls:

This readme is written by AI because I am very lazy
This project currently only collects Minecraft environment data and trains a segmentation model on it.
Further addition will be added if I'm in the mood

The mod jar is provided in this repository.

## What The Mod Does

The recorder captures 3 files per frame:

1. RGB image (`frame_xxxxx.png`)
2. Masked semantic image (`frame_xxxxx_masked.png`)
3. Metadata JSON (`frame_xxxxx.json`)

It writes data in this structure:

- [ai_data/PNG](ai_data/PNG)
- [ai_data/Masked](ai_data/Masked)
- [ai_data/Metadata](ai_data/Metadata)

And a shared label map file at root:

- [ai_data/block_labels.json](ai_data/block_labels.json)

Default output location:

- Singleplayer: `.minecraft/saves/<current_world>/ai_data`
- Multiplayer fallback: `.minecraft/ai_data`

## Mod Commands (5)

1. `/recordplayer <username>`
: choose which player to track for raycast labeling/metadata context.

2. `/recordframes <1-120>`
: set target capture rate in frames per second (triplets per second).

3. `/startrecord`
: start recording.

4. `/stoprecord`
: stop recording.

5. `/setoutputfolder <path>`
: override output directory.

## Mod Installation

1. Install Fabric Loader for Minecraft `1.21.3`.
2. Install Fabric API.
3. Put the mod jar and Fabric API jar in `.minecraft/mods`.
4. Launch Minecraft using the Fabric profile.

## Dataset Layout Used By Training

This repository expects:

- [Data/train_data](Data/train_data)
- [Data/val_data](Data/val_data)
- [Data/test_data](Data/test_data)

Each split should contain:

- `PNG/`
- `Masked/`
- `Metadata/`
- `block_labels.json`

## Current Folder Structure (This Repository)

Use this to verify paths if imports or file loading fail.
If your folder tree is different, then you might have to rewrite some part of the code in the training python file
```text
Ai Project/
	best_minecraft_model.pth
	Environment trainer.py
	Environment tester.py
	README.md
	Data/
		train_data/
			block_labels.json
			PNG/
			Masked/
			Metadata/
		val_data/
			block_labels.json
			PNG/
			Masked/
			Metadata/
		test_data/
			block_labels.json
			PNG/
			Masked/
			Metadata/
```

If you get path errors, check these first:

1. Run scripts from project root (`Ai Project/`).
2. Make sure split names are exactly `train_data`, `val_data`, `test_data`.
3. Make sure subfolder names are exactly `PNG`, `Masked`, `Metadata`.
4. Make sure `block_labels.json` exists in each split folder.
5. Make sure frame triplets are matched (same frame id across PNG/mask/json).

## Environment Trainer Overview

Main trainer script:

- [Environment trainer.py](Environment%20trainer.py)

### 1) Loading and Decoding

`EnvironmentDataset` loads RGB images, masked PNGs, and metadata JSON files.

Masked PNGs are RGB-encoded class IDs, decoded as:

$$
id = (R << 16) | (G << 8) | B
$$

So mask tensors become `H x W` integer class maps.

### 2) Semantic-to-Train Label Mapping

The trainer reads semantic labels from `block_labels.json` (`id_to_semantic` if present, else `id_to_block`).

It builds:

- global `semantic_to_train`
- global `train_to_semantic`
- a LUT for fast remapping

This guarantees stable class meaning across all images.

### 3) Ignore Index

Unknown/out-of-range labels are mapped to `IGNORE_INDEX = -1` and ignored by losses.

### 4) Model

The current baseline is `simpleUNet` (lightweight encoder/decoder segmentation model).

### 5) Loss Functions

Training uses a combined loss:

- Cross-Entropy
- Focal Loss

Combined as:

$$
L = 0.7 * CE + 0.3 * Focal
$$

This helps reduce collapse to dominant classes (like predicting mostly air).

### 6) Train / Validation / Test Flow

Per epoch:

1. Train on [Data/train_data](Data/train_data)
2. Validate on [Data/val_data](Data/val_data)
3. Save best checkpoint by validation loss

After training ends:

1. Evaluate once on [Data/test_data](Data/test_data)

### 7) Checkpoint Resume

Checkpoint file:

- [best_minecraft_model.pth](best_minecraft_model.pth)

If it exists, training resumes from saved:

- model weights
- optimizer state
- last epoch
- best validation loss

This prevents losing progress during long CPU training runs.

## Environment Tester

Evaluation helper script:

- [Environment tester.py](Environment%20tester.py)

Use it to test model behavior and inspect outputs without retraining.

