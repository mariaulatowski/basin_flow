from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class WyMonth:
    wy_month: int
    cal_year: int
    cal_month: int


def water_year_months(wy: int) -> list[WyMonth]:
    months: list[WyMonth] = []
    for wy_month, (year, month) in enumerate(
        [
            (wy - 1, 10),
            (wy - 1, 11),
            (wy - 1, 12),
            (wy, 1),
            (wy, 2),
            (wy, 3),
            (wy, 4),
            (wy, 5),
            (wy, 6),
            (wy, 7),
            (wy, 8),
            (wy, 9),
        ],
        start=1,
    ):
        months.append(WyMonth(wy_month=wy_month, cal_year=year, cal_month=month))
    return months


def build_filename_legacy(element: str, product: str, grid: str, year: int, month: int) -> str:
    return f"PRISM_{element}_{product}_{grid}_{year}{month:02d}_bil.zip"


def build_filename_timeseries(element: str, year: int, month: int, region: str, dataset: str) -> str:
    return f"prism_{element}_{region}_{dataset}_{year}{month:02d}.zip"


def build_url(base_url: str, filename: str, *, element: str, year: int, source_mode: str) -> str:
    base = base_url.rstrip("/")
    if source_mode == "legacy":
        return f"{base}/{element}/{year}/{filename}"
    if source_mode == "timeseries":
        return f"{base}/{element}/monthly/{year}/{filename}"
    raise ValueError(f"Unknown source_mode={source_mode}")


def download_file(url: str, out_path: Path, timeout: int = 120) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                return False
            with out_path.open("wb") as f:
                shutil.copyfileobj(response, f)
        return True
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except URLError:
        raise


def extract_zip(zip_path: Path, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            zf.extract(name, target_dir)
            extracted.append(target_dir / name)
    return extracted


def find_extracted_raster(extracted_dir: Path, element: str, year: int, month: int) -> Path | None:
    key = f"{element}_{year}{month:02d}"

    # Prefer GeoTIFF from PRISM's new COG packaging; fall back to BIL if present.
    preferred_suffixes = [".tif", ".bil"]
    for suffix in preferred_suffixes:
        matches = sorted(
            [
                p
                for p in extracted_dir.rglob("*")
                if p.is_file() and p.suffix.lower() == suffix and key in p.name.lower()
            ]
        )
        if matches:
            return matches[0]

    return None


def iter_elements(elements_csv: str) -> Iterable[str]:
    for token in elements_csv.split(","):
        e = token.strip().lower()
        if not e:
            continue
        if e not in {"ppt", "tmean"}:
            raise ValueError(f"Unsupported PRISM element '{e}'. Use ppt,tmean.")
        yield e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PRISM monthly grids for a target water year (Oct..Sep) "
            "and extract them for use with build_real_nlcd_prism_inputs.py"
        )
    )
    parser.add_argument("--wy", type=int, required=True, help="Water year (example: 2018)")
    parser.add_argument(
        "--elements",
        type=str,
        default="ppt,tmean",
        help="Comma-separated PRISM elements to download (default: ppt,tmean)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://ftp.prism.oregonstate.edu/time_series/us/an/4km",
        help="PRISM base URL (default uses 2025+ time_series structure)",
    )
    parser.add_argument(
        "--product",
        type=str,
        default="stable",
        help="Legacy mode only: PRISM product tier in filename (default: stable)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="4kmM3",
        help="Legacy mode only: PRISM grid code in filename (default: 4kmM3)",
    )
    parser.add_argument(
        "--source-mode",
        type=str,
        choices=["timeseries", "legacy"],
        default="timeseries",
        help="PRISM source layout (default: timeseries)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us",
        help="Time-series mode only: PRISM region code (default: us)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="25m",
        help="Time-series mode only: filename dataset token (default: 25m for 4km directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"c:\Users\mu3575\Documents\WAM\inputData\prism_monthly"),
        help="Output directory root (default: workspace inputData/prism_monthly)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if zip already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    months = water_year_months(args.wy)
    elements = list(iter_elements(args.elements))

    total_attempts = 0
    total_downloaded = 0
    total_missing = 0

    # Record discovered extracted monthly rasters so we can print exact patterns.
    sample_raster: dict[str, Path] = {}

    for element in elements:
        element_dir = args.out_dir / element
        zip_dir = element_dir / "zip"
        extracted_dir = element_dir / "extracted"

        for wm in months:
            if args.source_mode == "legacy":
                filename = build_filename_legacy(element, args.product, args.grid, wm.cal_year, wm.cal_month)
            else:
                filename = build_filename_timeseries(
                    element,
                    wm.cal_year,
                    wm.cal_month,
                    region=args.region,
                    dataset=args.dataset,
                )

            url = build_url(
                args.base_url,
                filename,
                element=element,
                year=wm.cal_year,
                source_mode=args.source_mode,
            )
            zip_path = zip_dir / filename

            total_attempts += 1

            if args.skip_existing and zip_path.exists():
                print(f"SKIP existing: {zip_path}")
            else:
                print(f"DOWNLOAD {url}")
                ok = download_file(url, zip_path)
                if not ok:
                    total_missing += 1
                    print(f"MISSING {url}")
                    continue
                total_downloaded += 1

            extracted = extract_zip(zip_path, extracted_dir)
            bil_count = len([p for p in extracted if p.suffix.lower() == ".bil"])
            tif_count = len([p for p in extracted if p.suffix.lower() == ".tif"])
            print(
                f"EXTRACTED {zip_path.name} -> {extracted_dir} "
                f"(tif_files={tif_count}, bil_files={bil_count})"
            )

            month_raster = find_extracted_raster(extracted_dir, element, wm.cal_year, wm.cal_month)
            if month_raster is not None and element not in sample_raster:
                sample_raster[element] = month_raster

            if not args.keep_zip and zip_path.exists():
                zip_path.unlink()

    print("PRISM_DOWNLOAD_COMPLETE")
    print(f"water_year={args.wy}")
    print(f"attempted={total_attempts} downloaded={total_downloaded} missing={total_missing}")

    if args.source_mode == "legacy":
        product = args.product
        grid = args.grid
        ppt_pattern = (
            args.out_dir / "ppt" / "extracted" / f"PRISM_ppt_{product}_{grid}_{{yyyy}}{{mm:02d}}_bil.bil"
        )
        tmean_pattern = (
            args.out_dir / "tmean" / "extracted" / f"PRISM_tmean_{product}_{grid}_{{yyyy}}{{mm:02d}}_bil.bil"
        )
    else:
        # Expected extracted time-series raster naming inside zip payload.
        # Pattern token remains {yyyy}{mm:02d} so downstream script can resolve by month.
        ppt_pattern = (
            args.out_dir
            / "ppt"
            / "extracted"
            / f"prism_ppt_{args.region}_{args.dataset}_{{yyyy}}{{mm:02d}}.tif"
        )
        tmean_pattern = (
            args.out_dir
            / "tmean"
            / "extracted"
            / f"prism_tmean_{args.region}_{args.dataset}_{{yyyy}}{{mm:02d}}.tif"
        )

    print("Use these patterns with build_real_nlcd_prism_inputs.py:")
    print(f"  --prism-ppt-pattern \"{ppt_pattern}\"")
    print(f"  --prism-tmean-pattern \"{tmean_pattern}\"")

    if sample_raster:
        print("Detected extracted sample rasters:")
        for element in sorted(sample_raster):
            print(f"  {element}: {sample_raster[element]}")

    # Non-zero exit if any monthly files were not found.
    if total_missing > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
