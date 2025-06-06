#!/usr/bin/env python3
import os
import gc
import json
import utm
import numpy as np
import xarray as xr
import dask

from obspy import UTCDateTime, read_inventory
from dask.distributed import Client, LocalCluster

from seismic_beamformer import SeismicBeamformer

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# USER PARAMETERS — edit these
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
RUN_START    = "2023-12-09T00:00:00"
RUN_END      = "2024-10-08T00:00:00"

MSEED_DIR     = "/media/chopp/Data1/chet-ingeneous/Gabbs_MSEED/"
INVENTORY_XML = "/media/chopp/Data1/chet-ingeneous/stations/Gabbs_Inventory.xml"
OUTDIR        = "/media/chopp/Data1/chet-ingeneous/Gabbs_Beamforming/"

SAMP_RATE   = 20.0      # Hz
CHUNK_S     = 1200.0    # for filter/CSD chunk‐size
WIN_S       = 50.0      # window length for CSD
OVERLAP     = 0.2
PAD_S       = 10.0

# ─── NEW: how we want the raw-Zarr to be chunked ───────────────
TRS_PER_CHK = 30                       # traces per chunk  (≈10 stations)
CHUNKS_T    = int(CHUNK_S * SAMP_RATE) # 1 200 s × 20 Hz → 24 000 samples

BAZS  = np.arange(0,360,5.0)
SLOWS = np.linspace(0.0,0.003,30)
BANDS = [(0.1,1.0),(1.0,5.0),(5.0,10.0)]

# how large to make each raw‐Zarr time chunk?
WRITE_CHUNK_S = 86400                  # 10 minutes
WRITE_CHUNK_N = int(WRITE_CHUNK_S * SAMP_RATE)

DASK_TMP = "/media/chopp/Data1/dask-spill"       # put it on a big disk
os.makedirs(DASK_TMP, exist_ok=True)

# ─── add somewhere near the top of the driver script ───────────────────

def raw_store_complete(path: str,
                       expected_ntr:  int,
                       expected_npts: int) -> bool:
    """
    True  -> folder exists and variable 'u' has
               shape   (expected_ntr,  expected_npts)  AND
               chunks  (all 1,        entire npts)
    False -> any other situation (missing folder, wrong size, etc.)
    """
    if not os.path.isdir(path):
        return False

    try:
        ds = xr.open_zarr(path, consolidated=True)
    except Exception:
        return False          # not a valid consolidated store

    if "u" not in ds:
        return False

    # ---- shape ---------------------------------------------------------
    if tuple(ds.u.shape) != (expected_ntr, expected_npts):
        return False

    tr_chunks, t_chunks = ds.u.chunks
    if sum(tr_chunks) != expected_ntr:
        return False
    if sum(t_chunks) != expected_npts:
        return False
    return True

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# helper: split into daily (or partial) blocks
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def make_daily_blocks(start_iso, end_iso):
    t0 = UTCDateTime(start_iso)
    t1 = UTCDateTime(end_iso)
    blocks = []
    cur = t0
    while cur < t1:
        nxt = cur + 86400
        end = nxt if nxt < t1 else t1
        blocks.append((cur, end))
        cur = end
    return blocks


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Ask Dask to start spilling way before a hard OOM
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
dask.config.set({
  "distributed.worker.memory.target":    0.6,
  "distributed.worker.memory.spill":     0.7,
  "distributed.worker.memory.pause":     0.8,
  "distributed.worker.memory.terminate":0.95,
})


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# STAGE 1: read MiniSEED → raw Zarr (one block per day)
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def stage1_write_raw(inv, blocks):
    stations      = {sta.code for net in inv for sta in net}
    expected_ntr  = len(stations) * 3            # 50×3 = 150
    expected_npts = int(86400 * SAMP_RATE)       # 1 728 000

    cluster = LocalCluster(
        n_workers=6,
        threads_per_worker=1,
        memory_limit="9GB",
        resources={"raw-io": 1}
    )
    client = Client(cluster)

    for t0, t1 in blocks:
        tag   = f"{t0.year}_{t0.julday:03d}"
        raw_z = os.path.join(OUTDIR, f"raw_{tag}.zarr")

        # skip if everything already present
        if raw_store_complete(raw_z, expected_ntr, expected_npts):
            print(f"✓  {raw_z} already complete – skipping")
            continue

        print(f"Writing {raw_z} …")
        bf = SeismicBeamformer(
                mseed_dir=MSEED_DIR, inventory=inv,
                t0=t0, t1=t1,
                samp_rate=SAMP_RATE,
                chunk_size_s=CHUNK_S,
                chunks_tr=TRS_PER_CHK,
                chunks_t=CHUNKS_T,
             )
        ds  = bf.load_data()
        ds2 = ds.chunk({"tr": TRS_PER_CHK, "time": CHUNKS_T})

        write = ds2.to_zarr(raw_z,
                            mode="w",
                            consolidated=True,
                            compute=False,
                            encoding={"u": {"compressor": None}})
        client.compute(write).result()

        del bf, ds, ds2
        gc.collect()

    client.close()
    cluster.close()

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# STAGE 2: open raw Zarr → preprocess → CSD → csd Zarr
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def stage2_compute_csd(inv, blocks):
    cluster = LocalCluster(
        n_workers         = 5,
        threads_per_worker= 1,
        memory_limit      = "12GB",
        local_directory   = DASK_TMP,   ### ←  NEW  ###
        resources         = {"csd-window": 1},
    )
    client = Client(cluster)

    chunks_t = int(CHUNK_S * SAMP_RATE)      # 1200 s × 20 Hz = 24000

    for (t0, t1) in blocks:
        tag   = f"{t0.year}_{t0.julday:03d}"
        raw_z = os.path.join(OUTDIR, f"raw_{tag}.zarr")
        csd_z = os.path.join(OUTDIR, f"csd_{tag}.zarr")

        # instantiate beamformer and point it at raw Zarr
        bf = SeismicBeamformer(
            mseed_dir=None,
            inventory=inv,
            t0=t0, t1=t1,
            samp_rate=SAMP_RATE,
            chunk_size_s=CHUNK_S,
            chunks_tr=TRS_PER_CHK,
            chunks_t=CHUNKS_T,
        )
        bf.raw_zarr = raw_z

        # open + rechunk one station at a time
        ds_raw    = xr.open_zarr(raw_z, consolidated=True)
        # ds_raw     = ds_raw.chunk({"tr": 1, "time": chunks_t})

        bf.ds = ds_raw

        bf.preprocess(
            band=(0.1,9.0),
            window_s=WIN_S,
            overlap=OVERLAP,
            pad_s=PAD_S
        )

        # driver‐side CSD Zarr spill
        bf.compute_csd_dask(spill_zarr=csd_z)

        del bf, ds_raw
        gc.collect()

    client.close()
    cluster.close()


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# STAGE 3: open CSD → beamforming → beam Zarr per band/day
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def stage3_beam(inv, blocks):
    cluster = LocalCluster(
        n_workers=5,
        threads_per_worker=1,
        memory_limit="12GB"
    )
    client = Client(cluster)

    for (t0, t1) in blocks:
        tag   = f"{t0.year}_{t0.julday:03d}"
        csd_z = os.path.join(OUTDIR, f"csd_{tag}.zarr")

        ds_c   = xr.open_zarr(csd_z, consolidated=True)
        Rf     = ds_c.Rf.data
        freqs  = ds_c.freq.values
        trlist = ds_c.tr.values.tolist()    # e.g. ["STA1-Z","STA1-N",...]

        # recompute UTM coords for each trace from Inventory
        coords = []
        for nm in trlist:
            sta = nm.split("-")[0]
            for net in inv:
                for st in net:
                    if st.code == sta:
                        e,n,_,_ = utm.from_latlon(st.latitude,
                                                  st.longitude)
                        coords.append((e,n))
                        break
                else:
                    continue
                break
        coords = np.array(coords)

        for (fmin,fmax) in BANDS:
            bz = os.path.join(
                OUTDIR,
                f"beam_{fmin:.2f}-{fmax:.2f}_{tag}.zarr"
            )

            # lightweight beam‐scan
            bf = SeismicBeamformer.__new__(SeismicBeamformer)
            bf.Rf     = Rf
            bf.freqs  = freqs
            bf.coords = coords
            bf.t0, bf.t1 = t0, t1

            bf.compute_beams(
                bazs=BAZS,
                slows=SLOWS,
                method="bartlett",
                freq_band=(fmin,fmax)
            )

            da = xr.DataArray(
                bf.P_fb,
                dims=("freq","beam"),
                coords={"freq":bf.freqs_sel,
                        "beam":np.arange(bf.nbaz*bf.nslow)},
            )
            dsb = da.to_dataset(name="P_fb")
            dsb.to_zarr(bz, mode="w", consolidated=True)

            del bf, dsb
            gc.collect()

    client.close()
    cluster.close()


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ENTRY POINT
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    inv    = read_inventory(INVENTORY_XML).select(station="G*")
    blocks = make_daily_blocks(RUN_START, RUN_END)

    # stage1_write_raw(inv,    blocks)
    stage2_compute_csd(inv, blocks)
    stage3_beam(inv,        blocks)

if __name__ == "__main__":
    main()