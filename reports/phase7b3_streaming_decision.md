# Phase 7B.3 Data-Access Redesign & Streaming Decision Report

### 1. Viable Strategy
The only scientifically valid method to execute the exact historical reduction without repeating remote downloads is **Temporal Chunking (Option B)**. This involves reading the exact Zarr chunk geometry `(time=1, lat=721, lon=1440)` sequentially, cropping the India subset per time step, and accumulating the 1991-2019 data into a local disk-backed Zarr store before running the percentile calculation locally.

### 2. Estimated Network Transfer
**175.8 GB**.
Every time step is bundled globally. Extracting 42,368 time steps requires downloading the full 4.15 MB chunk for every step.

### 3. Estimated Local Disk Requirement
**2.4 GB**.
The extracted India domain `(lat=117, lon=121)` for 42,368 time steps takes exactly 2.4 GB of uncompressed local disk space.

### 4. Estimated Runtime
**~12 to 18 Hours**.
Streaming 42,368 separate HTTP requests over a residential or standard internet connection to a GCS bucket inherently suffers from latency overhead (typically 1-2 seconds per chunk download and slice).

### 5. Peak RAM
**>86.2% (Abort Triggered)**.
The baseline RAM of this laptop is idling above 86%. Attempting to open the metadata and instantiate the very first lazy chunk immediately breached the mandatory 85% Hard Abort limit. The benchmark correctly and safely aborted.

### 6. Whether Exact Percentile is Feasible
It is methodologically feasible, but computationally blocked on this specific machine due to the combination of the high baseline RAM and the 176 GB network download penalty.

### 7. Whether Approximate Percentile is Acceptable
An approximate percentile (like T-Digest) would alleviate RAM usage during accumulation, but it **does not solve the 176 GB network bottleneck or the 85% baseline RAM constraint**. Therefore, changing the scientific methodology to an approximate method is unwarranted, as it does not rescue the computation.

### 8. Recommended Implementation
Perform this exact reduction on a **Cloud Compute Instance** (e.g., Google Colab or a GCP VM) located in the same geographic region as the WeatherBench2 GCS bucket. A cloud node with sufficient RAM can process the 176 GB temporally in minutes rather than hours, outputting the final 14 MB climatology threshold artifact for local download.

### 9. Whether Full 1991-2019 Computation is Practical on this Machine
**Not Practical.**
The WeatherBench2 Zarr layout (global spatial chunks) makes fetching localized temporal timeseries fundamentally inefficient over remote networks on memory-constrained hardware.

## Final Status
**PHASE 7B.3: NETWORK BOTTLENECK**
