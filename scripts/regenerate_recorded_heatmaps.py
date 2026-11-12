"""Regenerate one existing ToF-Doppler session using read-only valder modules.

Keeps the existing manifest's frame identities, PNG names and dimensions. Stage
and validate first; --install backs up every replaced file inside this repo.
Run with the remote sensing environment; no files are written into valder.
"""
import os
import sys

sys.dont_write_bytecode = True
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import csv
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path


GIB = 1024 ** 3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resources(path):
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    available, disk = mem["MemAvailable"], shutil.disk_usage(path).free
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if available < 3 * GIB or disk < 5 * GIB or rss > 2 * GIB:
        raise RuntimeError(f"Resource reserve reached: available={available}, disk={disk}, peak_rss={rss}")
    return {"available_ram_gib": round(available / GIB, 3),
            "free_disk_gib": round(disk / GIB, 3),
            "peak_rss_mib": round(rss / 1024 ** 2, 1)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstc-root", type=Path, default=Path.home() / "Projects/NSTC")
    parser.add_argument("--session", required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if Path(args.session).name != args.session or args.session in (".", ".."):
        parser.error("Supply one session directory name")
    root = args.nstc_root.resolve()
    repo = root / "baf_roi_history6dof_v2"
    data = root / "sensor-agent-publish-subscribe"
    source = root / "valder"
    # A hard address-space ceiling complements checks before and between frames.
    resource.setrlimit(resource.RLIMIT_AS, (6 * GIB, 6 * GIB))
    print("START_RESOURCES", resources(repo), flush=True)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.io import loadmat, savemat

    sys.path.insert(0, str(source / "rt_data_processing"))
    import csi_merger
    import tof_doppler_generator as generator

    output = data / "heatmaps/heatmap_result"
    figures_rel = Path("figures") / args.session / "ToF-Doppler"
    mat_rel = Path("mat") / args.session / "ToF-Doppler/smoothed_CSI_avg.mat"
    figures = output / figures_rel
    fields, manifest = read_csv(figures / "heatmap_timestamp_manifest.csv")
    if not manifest:
        raise ValueError("Empty manifest")
    if [int(row["heatmap_index"]) for row in manifest] != list(range(len(manifest))):
        raise ValueError("Manifest heatmap indices must be contiguous")
    expected_png = {row["filename"] for row in manifest}
    if len(expected_png) != len(manifest) or expected_png != {p.name for p in figures.glob("*.png")}:
        raise ValueError("Manifest and existing PNG set differ")
    for row in manifest:
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("Unsafe output filename")
        if Path(row["filepath"]) != figures / row["filename"]:
            raise ValueError("Unexpected manifest output path")
    metadata = json.loads((figures / "heatmap_metadata.json").read_text())
    if any(row["normalization"] != "global" for row in manifest):
        raise ValueError("This adapter currently supports existing global normalization")
    if metadata.get("quantity", "power_db") != "power_db":
        raise ValueError("This adapter expects existing dB images")
    originals = sorted(p for p in figures.iterdir() if p.is_file()) + [output / mat_rel]
    before = {str(p.relative_to(output)): sha(p) for p in originals}
    job = repo / "runs/heatmap_regeneration" / args.session / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stage = job / "staged"
    (stage / figures_rel).mkdir(parents=True, exist_ok=False)
    (stage / mat_rel).parent.mkdir(parents=True)
    write_json(job / "original_sha256.json", before)

    # Legacy CSI indices refer to sorted NIC1 sidecars, not to a newly merged
    # union of all NIC timelines. Keep that exact timeline and match by packet ID.
    reference_files = sorted((data / "artifacts" / args.session / "arrays/csi.rx.1").glob("*.npy"))
    indices = np.asarray([int(row["csi_index"]) for row in manifest], dtype=np.int64)
    for row, idx in zip(manifest, indices):
        if idx >= len(reference_files) or reference_files[idx].stem != row["csi_timestamp"]:
            raise ValueError("Legacy manifest does not match NIC1 sidecar timestamps")
    nic_rows = []
    for nic in range(1, 5):
        rows = []
        for path in sorted((data / "db" / args.session / f"csi.rx.{nic}").glob("*.csv")):
            rows.extend(read_csv(path)[1])
        mapping = {}
        for row in rows:
            key = (int(row["packet_seq"]), int(row["packet_taskId"]))
            if key not in mapping or int(row["rx_system_ns"]) < int(mapping[key]["rx_system_ns"]):
                mapping[key] = row
        nic_rows.append((rows, mapping))
    by_filename = {Path(row["array_saved"]).name: row for row in nic_rows[0][0]}
    reference = [by_filename[p.name] for p in reference_files]
    sample = csi_merger.load_npy_file(reference_files[len(reference_files) // 2], 2025)
    subcarriers = sample.shape[-1]
    if subcarriers not in (2025, 1001):
        raise ValueError(f"Unrecognized radio bandwidth: {subcarriers} subcarriers")
    step = generator.subcarrier_step(subcarriers)
    count_sc = len(range(0, subcarriers, step))
    # Slice before stacking/interpolating: ~4 MB rather than ~120 MB for this session.
    csi = np.empty((len(reference), 2, 8, count_sc), dtype=np.complex64)
    missing = {}
    ordering_checks = {}
    for nic, (_, mapping) in enumerate(nic_rows, 1):
        block = np.full((len(reference), 2, 2, count_sc), np.nan + 1j * np.nan, dtype=np.complex64)
        missing[str(nic)] = 0
        correlations = []
        for i, row in enumerate(reference):
            match = mapping.get((int(row["packet_seq"]), int(row["packet_taskId"])))
            path = None if match is None else data / "artifacts" / args.session / "arrays" / f"csi.rx.{nic}" / Path(match["array_saved"]).name
            if path is None or not path.is_file():
                missing[str(nic)] += 1
                continue
            packet = csi_merger.load_npy_file(path, subcarriers)
            if i in (0, len(reference) // 2, len(reference) - 1):
                amp = np.abs(packet[0, 0]); amp -= amp.mean()
                correlation = float(amp[:-1] @ amp[1:] / (amp @ amp))
                if not np.isfinite(correlation) or correlation <= .5:
                    raise ValueError(f"NIC {nic}: legacy CSI ordering check failed")
                correlations.append(correlation)
            block[i] = packet[..., ::step]
        if missing[str(nic)] > len(reference) * .05:
            raise ValueError(f"NIC {nic}: more than 5% missing packets")
        for tx in range(2):
            for rx in range(2):
                for sc in range(count_sc):
                    series = block[:, tx, rx, sc]
                    block[:, tx, rx, sc] = csi_merger.interp_timeseries(series.real, series.imag)
        csi[:, :, 2 * (nic - 1):2 * nic] = block
        ordering_checks[str(nic)] = correlations
    del block
    antenna_order = [0, 1, 2, 3, 6, 7, 4, 5] if subcarriers == 2025 else [6, 7, 4, 5, 2, 3, 0, 1]
    csi = csi[:, :, antenna_order]
    # Verify the early subcarrier slice against valder's original full-band
    # load_amplitude on a fixture with identical retained subcarriers.
    fixture = job / "amplitude_equivalence_fixture.npz"
    full = np.zeros((60, 2, 8, subcarriers), dtype=np.complex64)
    full[..., ::step] = csi[:60]
    np.savez(fixture, csi=full)
    del full
    expected_amp, expected_step = generator.load_amplitude(fixture)
    reduced = np.abs(csi[:60].astype(np.complex128))
    reduced /= np.maximum(reduced.mean(-1, keepdims=True), 1e-9)
    np.testing.assert_allclose(generator.remove_background(reduced), expected_amp, rtol=1e-12, atol=1e-12)
    assert step == expected_step
    del expected_amp, reduced
    fixture.unlink()
    amp = np.abs(csi.astype(np.complex128))
    del csi
    amp /= np.maximum(amp.mean(-1, keepdims=True), 1e-9)
    amp = generator.remove_background(amp)
    tofdop = generator.build_tof_dop(source / "CSI-MUSIC-heatmap-baseline", count_sc, step)
    spectra = np.empty((len(manifest), len(generator.TAU_AXIS), len(generator.DOPPLER_AXIS)), dtype=np.float64)
    started = time.monotonic()
    for i, idx in enumerate(indices):
        resources(repo)
        spectra[i] = 10 * np.log10(np.maximum(tofdop.cal_spectrum(tofdop.Rxx_smooth(amp, int(idx)))[2], 1e-12))
        if i % 25 == 0 or i + 1 == len(manifest):
            print("FRAME", i + 1, "/", len(manifest), "elapsed_s", round(time.monotonic() - started, 1), resources(repo), flush=True)
    if not np.isfinite(spectra).all():
        raise ValueError("Non-finite spectrum")
    vmin, vmax = float(spectra.min()), float(spectra.max())
    source_hashes = {str(p.relative_to(root)): sha(p) for p in [
        source / "rt_data_processing/csi_merger.py", source / "rt_data_processing/tof_doppler_generator.py",
        source / "CSI-MUSIC-heatmap-baseline/MUSIC.py", Path(__file__).resolve()]}
    details = {"generator": "valder.rt_data_processing.tof_doppler_generator", "source_sha256": source_hashes,
        "settings": generator.TOF_DOP_SETTING, "subcarriers": subcarriers, "subcarrier_step": step,
        "antenna_order": antenna_order, "missing_packets_by_nic": missing, "ordering_checks": ordering_checks,
        "timeline": "Original manifest and sorted NIC1 sidecars; other NICs joined by packet_seq/packet_taskId",
        "preprocess": "valder gain-normalized amplitude minus centered 50-frame mean",
        "past_only": False, "startup_and_end": "valder centered MUSIC windows clamped to recording boundaries",
        "physical_interpretation": "Amplitude-domain MUSIC pseudo-spectrum; not calibrated coherent CSI",
        "frame_shape": list(spectra.shape[1:]), "tau_s": generator.TAU_AXIS.tolist(),
        "doppler_hz": generator.DOPPLER_AXIS.tolist(), "flatten_order": "C (tau, doppler)",
        "vmin": vmin, "vmax": vmax, "amplitude_equivalence": "PASS, rtol=atol=1e-12"}
    old_mat = loadmat(output / mat_rel)
    savemat(stage / mat_rel, {"idx": old_mat["idx"], "spectrum": spectra.reshape(len(manifest), -1),
        "tau_s": generator.TAU_AXIS, "doppler_hz": generator.DOPPLER_AXIS,
        "csi_index": indices, "metadata_json": json.dumps(details)}, do_compression=True)
    for i, row in enumerate(manifest):
        if i % 50 == 0:
            resources(repo)
        width, height = int(row["width"]), int(row["height"])
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.imshow(spectra[i], aspect="auto", origin="lower", cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.savefig(stage / figures_rel / row["filename"], dpi=100, pad_inches=0)
        plt.close(fig)
        row["vmin"], row["vmax"] = str(vmin), str(vmax)
    with (stage / figures_rel / "heatmap_timestamp_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(manifest)
    metadata.update(total_heatmaps=len(manifest), total_csi_timestamps=len(reference), processed=len(manifest), failed=0,
                    quantity="power_db", normalization="global", regeneration=details)
    write_json(stage / figures_rel / "heatmap_metadata.json", metadata)
    # A staging tree must have exactly the old file set, with valid image sizes,
    # identical row identities and identical legacy MAT indices before installation.
    after = {str(p.relative_to(stage)): sha(p) for p in stage.rglob("*") if p.is_file()}
    if set(after) != set(before):
        raise ValueError("Replacement file set differs from the original")
    _, original_manifest = read_csv(figures / "heatmap_timestamp_manifest.csv")
    for old, new in zip(original_manifest, manifest):
        for key in fields:
            if key not in ("vmin", "vmax") and old[key] != new[key]:
                raise ValueError(f"Manifest identity changed: {key}")
        with Image.open(stage / figures_rel / new["filename"]) as image:
            assert image.size == (int(new["width"]), int(new["height"]))
            image.verify()
    check = loadmat(stage / mat_rel)
    np.testing.assert_array_equal(check["idx"], old_mat["idx"])
    np.testing.assert_array_equal(check["spectrum"], spectra.reshape(len(manifest), -1))
    report = {"session": args.session, "job": str(job), "installed": False, "frames": len(manifest),
        "original_spectrum_shape": list(old_mat["spectrum"].shape), "new_spectrum_shape": list(check["spectrum"].shape),
        "validation": "PASS: filenames, paths, timestamps, indices, PNG dimensions, finite MAT values, valder amplitude equivalence",
        "resources": resources(repo), "details": details}
    write_json(job / "replacement_sha256.json", after)
    write_json(job / "report.json", report)
    if args.install:
        resources(repo)
        if any(sha(output / rel) != digest for rel, digest in before.items()):
            raise ValueError("Original output changed during generation; refusing replacement")
        backup = job / "backup"
        for rel in before:
            dest = backup / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / rel, dest)
            if sha(dest) != before[rel]:
                raise ValueError("Backup checksum failed")
        try:
            for rel in after:
                os.replace(stage / rel, output / rel)
            if any(sha(output / rel) != digest for rel, digest in after.items()):
                raise ValueError("Installed checksum failed")
        except BaseException:
            for rel in before:
                shutil.copy2(backup / rel, output / rel)
            raise
        report.update(installed=True, backup=str(backup), resources=resources(repo))
        write_json(job / "report.json", report)
    print("COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
