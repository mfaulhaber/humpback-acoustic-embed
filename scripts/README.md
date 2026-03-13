# convert_audio_to_flac

Convert `.wav` and `.mp3` files to sibling `.flac` files without deleting the source files.

Basic usage:

```bash
uv run python scripts/convert_audio_to_flac.py /path/to/file.wav
uv run python scripts/convert_audio_to_flac.py /path/to/folder --verify-samples
```

Notes:

* File inputs are converted directly; directory inputs are scanned recursively.
* Existing `.flac` files are skipped.
* Existing sibling targets such as `clip.flac` next to `clip.wav` are skipped.
* `--verify-samples` decodes the source and output files and requires matching sample
  rate, sample count, and `max_abs_error <= 5e-5`. Failed verification deletes the
  newly written `.flac` file and returns a non-zero exit status.

---

# stage_s3_epoch_cache

Stage time-windowed data from **epoch-timestamped S3 directory structures** into a **local NVMe cache** using `s5cmd`.

This tool is designed for datasets organized with **Unix epoch timestamps as directory names**, for example:

```
s3://audio-orcasound-net/rpi_north_sjc/hls/1752303617/
```

It allows you to download **only the directories that overlap a requested UTC time range**, which is useful when working with large audio archives where the full dataset cannot be mirrored locally.

Typical use cases:

* Hydrophone or sensor recordings
* Time-windowed ML training datasets
* Staging subsets of very large S3 archives
* Building **local caches on GPU compute nodes**

The script:

1. Converts a requested time range into **epoch seconds**
2. Lists timestamp directories under a given S3 prefix
3. Uses a coarse epoch match, then refines using object `LastModified` overlap
4. Generates an **s5cmd command file**
5. Executes downloads in **parallel** by default (or can run in dry-run mode)

This approach avoids scanning the entire S3 dataset while maximizing throughput.

---

# Features

* Designed for **public S3 buckets**
* Uses `--no-sign-request` (no AWS credentials required)
* Automatically runs `s5cmd` with `--no-sign-request`
* Handles coarse/misaligned epoch partitions via object-overlap refinement
* High-performance downloads using **s5cmd**
* Prefix-level progress bar during downloads
* Generates reproducible command manifests
* Supports **resume behavior** via `--skip-existing`
* Supports plan-only mode via `--dry-run`
* Prefix discovery uses `list-objects-v2` with `start-after` and end-boundary early stop for faster narrow-window startup
* Compatible with large **NVMe staging volumes**

---

# Requirements

External tools required:

```
aws-cli
s5cmd
```

---

# Installation

## macOS

The easiest method on macOS is **Homebrew**.

### Install Homebrew (if needed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install dependencies

```bash
brew install awscli s5cmd
```

Verify installation:

```bash
aws --version
s5cmd version
```

---

## Ubuntu / Linux

Install dependencies:

```bash
apt update
apt install -y awscli wget
```

Install `s5cmd`:

```bash
wget https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz
tar -xzf s5cmd_2.3.0_Linux-64bit.tar.gz
mv s5cmd /usr/local/bin/
```

Verify:

```bash
aws --version
s5cmd version
```

---

# Python Environment (uv)

This project uses **uv** as the Python package manager.

Install `uv`:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:

```bash
uv venv
```

Install dependencies (if present):

```bash
uv pip install -r requirements.txt
```

Activate the environment:

```bash
source .venv/bin/activate
```

Run the script:

```bash
python stage_s3_epoch_cache.py --help
```

Because the script relies primarily on system tools (`aws` and `s5cmd`), Python dependencies are minimal.

---

# Expected S3 Layout

The script assumes timestamp directories under a prefix:

```
s3://audio-orcasound-net/
  rpi_north_sjc/
    hls/
      1752217217/
      1752303617/
      1752390017/
```

Each directory contains files such as:

```
segment_000.ts
segment_001.ts
playlist.m3u8
```

The numeric directory name represents a **Unix epoch timestamp**.

Some datasets use coarse partitions (for example, daily directories). In that case, the
directory timestamp itself may not fall inside a short query window, so the script
also checks object `LastModified` timestamps inside candidate directories.

---

# Local Cache Layout

The local cache mirrors the S3 structure:

```
/workspace/data_cache/
  rpi_north_sjc/
    hls/
      1752303617/
      1752303677/
```

This makes the cache **idempotent and resumable**.

---

# Basic Usage

Example: stage **24 hours of data**

```bash
uv run python stage_s3_epoch_cache.py \
  --bucket audio-orcasound-net \
  --prefix rpi_north_sjc/hls \
  --start "2025-07-12T00:00:00Z" \
  --hours 24 \
  --local-root /workspace/data_cache
```

This will:

1. Discover matching timestamp directories
2. Generate an `s5cmd` command file
3. Download matching directories into the local cache (with a per-prefix progress bar)

---

# Input Arguments

## Required

* `--bucket`: S3 bucket name
* `--prefix`: Root prefix containing epoch directories (for example `rpi_north_sjc/hls`)
* `--start`: Inclusive UTC start datetime (ISO-8601, for example `2025-07-12T00:00:00Z`)
* One of:
  * `--end`: Exclusive UTC end datetime
  * `--hours`: Positive integer number of hours after `--start`

## Optional

* `--local-root` (default: `/workspace/data_cache`): local cache root path
* `--commands-file`: output path for generated s5cmd commands
* `--matched-prefixes-file`: output path for matched epoch prefixes
* `--region` (default: `us-west-2`): AWS region for `aws` CLI listing calls
* `--dry-run`: generate manifests and print summary only; skip all downloads
* `--numworkers` (default: `64`): s5cmd worker count for each prefix copy command
* `--skip-existing`: skip prefixes whose local directory is already non-empty
* `--log-file`: optional path to write s5cmd output when execution is enabled
* `--pre-count` / `--no-pre-count` (default: `--pre-count`): enable/disable pre-count pass for object/file totals used by progress calculation

## Execution behavior

* Downloads run by default when `--dry-run` is not provided.
* Manifest files are always generated:
  * `matched_prefixes_<prefix>.txt`
  * `commands_<prefix>.txt`
* By default, a pre-count pass estimates total files/bytes across selected prefixes and uses those fixed totals as the progress denominator.
* Download-time progress is rendered by the script (tqdm) from structured `s5cmd --json` copy events, so totals do not grow during transfer.
* Use `--dry-run` to validate selection and manifests without downloading; generated copy commands are also printed to the console.

For `audio-orcasound-net/rpi_north_sjc/hls`, a request like
`2025-07-12T00:00:00Z` to `2025-07-12T01:00:00Z` can legitimately match the
directory `1752303617/` because object times (not just directory name) are used.

---

# Explicit Time Window

Instead of `--hours`, you can specify an end time.

```bash
uv run python stage_s3_epoch_cache.py \
  --bucket audio-orcasound-net \
  --prefix rpi_north_sjc/hls \
  --start "2025-07-12T00:00:00Z" \
  --end "2025-07-13T00:00:00Z" \
  --local-root /workspace/data_cache
```

---

# Resume Downloads

To avoid re-downloading cached directories:

```
--skip-existing
```

Example:

```bash
uv run python stage_s3_epoch_cache.py \
  --bucket audio-orcasound-net \
  --prefix rpi_north_sjc/hls \
  --start "2025-07-12T00:00:00Z" \
  --hours 24 \
  --skip-existing
```

---

# Parallel Download Performance

`s5cmd` uses multiple workers for parallel downloads.

Default:

```
--numworkers 64
```

You can increase this on high-bandwidth systems:

```
--numworkers 128
```

Example:

```bash
python stage_s3_epoch_cache.py \
  --bucket audio-orcasound-net \
  --prefix rpi_north_sjc/hls \
  --start "2025-07-12T00:00:00Z" \
  --hours 24 \
  --numworkers 128
```

---

# Dry Run Example

Use `--dry-run` to generate matched-prefix and command manifests without executing downloads:

```bash
uv run python stage_s3_epoch_cache.py \
  --bucket audio-orcasound-net \
  --prefix rpi_north_sjc/hls \
  --start "2025-07-12T00:00:00Z" \
  --hours 24 \
  --local-root /workspace/data_cache \
  --dry-run
```

---

# Generated Files

The script generates two files in the cache directory:

```
commands_<prefix>.txt
matched_prefixes_<prefix>.txt
```

Example:

```
commands_rpi_north_sjc_hls.txt
matched_prefixes_rpi_north_sjc_hls.txt
```

These provide:

* reproducible downloads
* auditability
* manual execution if needed

Manual execution example:

```bash
s5cmd --no-sign-request run commands_rpi_north_sjc_hls.txt
```

---

# Typical Workflow (GPU Node)

Recommended pipeline when working on GPU compute nodes:

```
S3 dataset
     ↓
stage subset with stage_s3_epoch_cache
     ↓
local NVMe cache
     ↓
TensorFlow / PyTorch pipeline
```

This avoids repeated S3 reads and significantly improves throughput.

---

# Limitations

Downloads are **prefix-level**. If any object in a matched prefix overlaps the requested
window, the script downloads the entire prefix directory.

Object-level filtering is not implemented in this tool.

---

# License

MIT
