# Implementation notes

## Model and spatial encoding

| Item | Retained implementation |
|---|---|
| Model input | float32 `[1,20,3,144,192]`, BGR/255 |
| Model output | `[1,20,1,144,192]`; final timestep supplies the live saliency map |
| Temporal history | causal, stride 8 by default; early frames repeat frame 0 |
| Tile grid | 5×9 logical regions, with aligned boundaries |
| Saliency smoothing | 0.7 previous score + 0.3 current score |
| Final policy | top six −2, next six −1, other 33 zero |
| QP map blocks | 16 pixels H.264; 32 H.265; 64 AV1 |
| Encoding | one panoramic coded picture per frame |
| Model SHA-256 | `51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa` |

The final policy is recovered from `SATC_FINAL_QUALITY_EXP09`, not inferred from
older tuning scripts. Runtime source is unpacked into ordinary Python/C++ files.
The compressed ONNX model keeps the repository small and is checked when unpacked.

## Differences to resolve when matching the manuscript

The manuscript describes 20 consecutive RGB frames. The retained deployment uses
BGR/255 and a stride-8 causal window. `satc.py encode --history-stride 1` changes
temporal sampling explicitly; it does not silently redefine historical results.

The manuscript applies cosine latitude weighting at logical tile centers. The
retained model-to-tile projection averages pixel latitude weights over logical
regions. These details should be reconciled before claiming byte-for-byte or
numerically identical reproduction of a paper configuration.

The final quality source and historical network/throughput source have different
spatial policies. The documentation identifies each. No historical result table
is republished as new measurements of the final policy.

The Ubuntu sender and upstream Quest receiver are assembled from separate source
snapshots. Their presence does not establish a newly tested integrated 4K60 run.
A completed H.264 encode can be replayed through the existing FIFO transport path;
a live all-codec final-policy encoder/controller/Quest launcher remains a separate
integration step.

## Provenance and validation

The source manifest records selected original file paths and hashes. Path and
entry-point changes are limited to portable configuration, readable source
materialization, and command wrappers. The model weights are unchanged.

Hardware-independent checks cover model integrity, final ranking/tie behavior,
codec block-map geometry, stream accounting, final matrix dimensions, and source
syntax. CUDA execution, native compilation, transport negotiation, and Quest
presentation were not executed in the remote build environment.
