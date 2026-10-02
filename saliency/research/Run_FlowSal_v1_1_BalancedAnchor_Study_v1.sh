#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/D-SAV360"
BASE_EXPECTED_SHA256="2b201b79cb2e841c5315ffbb3c8dfda3d2a970a4a0d4d7358439694100bd491b"

SEEDS_STRING="${SEEDS:-20260727 20260728 20260729}"
TASK_STEPS_STRING="${TASK_STEPS_SET:-250 500 750 1000}"

PAIR_EPOCHS="${PAIR_EPOCHS:-5}"
PAIR_BATCH="${PAIR_BATCH:-64}"
PAIR_LR="${PAIR_LR:-0.001}"
TASK_LR="${TASK_LR:-0.00005}"
MOTION_REG_WEIGHT="${MOTION_REG_WEIGHT:-0.05}"

MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
MAX_VAL_PAIRS="${MAX_VAL_PAIRS:-0}"
MAX_VAL_WINDOWS="${MAX_VAL_WINDOWS:-0}"

WARMUPS="${WARMUPS:-10}"
REPEATS="${REPEATS:-100}"

EXP_ROOT="$ROOT/flowsal_v1_1_balanced_anchor_study"
PROBE_ROOT="$ROOT/flowsal_v1_1_balanced_anchor_probe"
MODEL_ROOT="$EXP_ROOT/models"

STAMP="$(date +%F_%H-%M-%S)"
STUDY_DIR="$EXP_ROOT/results/$STAMP"
CONSOLE_LOG="$STUDY_DIR/console.log"

mkdir -p "$STUDY_DIR" "$MODEL_ROOT"

finish_shell() {
    status=$?
    echo
    echo "============================================================"
    echo "FlowSal v1.1 study exit status: $status"
    echo "Study directory: $STUDY_DIR"
    echo "Console log: $CONSOLE_LOG"
    echo "============================================================"
    echo "The terminal remains open. Type exit when finished."
    exec bash -i
}
trap finish_shell EXIT

{
    echo "===== FLOWSAL v1.1 BALANCED-ANCHOR STUDY ====="
    echo "Architecture: FlowSal-R192-T20-C8"
    echo "SST-Sal core trainable: NO"
    echo "Motion-stem parameters trainable: 811"
    echo "Runtime input: RGB only"
    echo "SEA-RAFT executed during training: NO"
    echo "SEA-RAFT-derived cached representation used: YES"
    echo "D-SAV360 final test opened: NO"
    echo "Seeds: $SEEDS_STRING"
    echo "Task checkpoints: $TASK_STEPS_STRING"
    echo "Task LR: $TASK_LR"
    echo "Motion regularization: $MOTION_REG_WEIGHT"
    echo

    BASE_SCRIPT="$(
        find "$HOME/Downloads" \
            -maxdepth 1 \
            -type f \
            -name 'Run_Unified_RGB_MotionStem_FrozenCore_Probe_v1*.sh' \
            -printf '%T@ %p\n' |
        sort -nr |
        head -n 1 |
        cut -d' ' -f2-
    )"

    if [[ -z "$BASE_SCRIPT" || ! -f "$BASE_SCRIPT" ]]; then
        echo "ERROR: The successful frozen-core probe script was not found."
        exit 1
    fi

    echo "Base script: $BASE_SCRIPT"
    echo "$BASE_EXPECTED_SHA256  $BASE_SCRIPT" | sha256sum -c -

    CHILD_SCRIPT="$STUDY_DIR/flowsal_v1_1_probe_noninteractive.sh"

    sed \
        's/^[[:space:]]*exec bash -i[[:space:]]*$/    exit "$status"/' \
        "$BASE_SCRIPT" \
        > "$CHILD_SCRIPT"

    chmod +x "$CHILD_SCRIPT"

    export CHILD_SCRIPT

    /home/mininet-ovs/venvs/satc/bin/python <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

path = Path(os.environ["CHILD_SCRIPT"])
text = path.read_text(encoding="utf-8")

replacements = {
    'EXP_ROOT="$ROOT/unified_rgb_motionstem_frozen_core_probe"':
        'EXP_ROOT="$ROOT/flowsal_v1_1_balanced_anchor_probe"',

    'echo "===== UNIFIED RGB MOTION-STEM FROZEN-CORE FEASIBILITY PROBE ====="':
        'echo "===== FLOWSAL v1.1 BALANCED-ANCHOR PROBE ====="',

    'UNIFIED_META = MODEL_ROOT / "Unified_RGB_Motion48_C8_frozen_core_probe.json"':
        'UNIFIED_META = MODEL_ROOT / "FlowSal_R192_T20_C8_v1_1.json"',

    'UNIFIED_ONNX = MODEL_ROOT / "Unified_RGB_Motion48_C8_frozen_core_static_opset16.onnx"':
        'UNIFIED_ONNX = MODEL_ROOT / "FlowSal_R192_T20_C8_v1_1_static_opset16.onnx"',

    'build_window_loader("train", 4096, shuffle=True)':
        'build_window_loader("train", 0, shuffle=True)',

    '"architecture": "unified_rgb_motion48_c8_frozen_core_probe"':
        '"architecture": "flowsal_r192_t20_c8_v1_1_balanced_anchor"',
}

for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"Required source text was not found:\n{old}")
    text = text.replace(old, new)

loss_pattern = re.compile(
    r"def composite_saliency_loss"
    r"\(prediction: torch\.Tensor, target: torch\.Tensor\):"
    r".*?"
    r"\n\n\ndef train_pair_stage",
    flags=re.DOTALL,
)

loss_replacement = r'''def composite_saliency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    """
    Distribution-balanced saliency loss.

    KLD and SIM supervise probability-map overlap.
    CC supervises spatial correlation.
    MSE receives a stronger weight to prevent the calibration
    degradation observed in FlowSal v1.
    """
    prediction_norm = minmax_normalize(prediction)
    target_norm = minmax_normalize(target)

    prediction_prob = probability_map(prediction)
    target_prob = probability_map(target)

    kld = torch.sum(
        target_prob
        * (
            torch.log(target_prob + 1e-8)
            - torch.log(prediction_prob + 1e-8)
        ),
        dim=(-2, -1),
    ).mean()

    cc_loss = (
        1.0
        - cc_value(
            prediction_norm,
            target_norm,
        )
    ).mean()

    sim = torch.minimum(
        prediction_prob,
        target_prob,
    ).sum(dim=(-2, -1)).mean()
    sim_loss = 1.0 - sim

    mse = F.mse_loss(
        prediction_norm,
        target_norm,
    )

    total = (
        kld
        + 0.50 * cc_loss
        + 0.25 * sim_loss
        + 10.0 * mse
    )

    return total, {
        "kld": kld,
        "cc_loss": cc_loss,
        "sim_loss": sim_loss,
        "mse": mse,
    }


def flowsal_task_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    kd_reference: torch.Tensor,
):
    """
    FlowSal v1.1 task objective.

    50% ground-truth saliency supervision
    40% frozen KD+SEA-RAFT output anchoring
    10% original teacher-map supervision
    """
    teacher_target = targets[:, 0]
    ground_truth = targets[:, 1]

    ground_truth_loss, gt_parts = composite_saliency_loss(
        prediction,
        ground_truth,
    )
    anchor_loss, anchor_parts = composite_saliency_loss(
        prediction,
        kd_reference,
    )
    teacher_loss, teacher_parts = composite_saliency_loss(
        prediction,
        teacher_target,
    )

    total = (
        0.50 * ground_truth_loss
        + 0.40 * anchor_loss
        + 0.10 * teacher_loss
    )

    return total, {
        "ground_truth": ground_truth_loss,
        "kd_anchor": anchor_loss,
        "teacher": teacher_loss,
        "ground_truth_parts": gt_parts,
        "anchor_parts": anchor_parts,
        "teacher_parts": teacher_parts,
    }


def train_pair_stage'''

text, count = loss_pattern.subn(
    loss_replacement,
    text,
    count=1,
)

if count != 1:
    raise RuntimeError(
        "Could not replace the original saliency-loss block."
    )

task_pattern = re.compile(
    r'''        optimizer\.zero_grad\(set_to_none=True\)
        with torch\.amp\.autocast\("cuda", dtype=torch\.float16, enabled=True\):
            motion_full, prediction_low_sequence = unified\.infer_motion\(rgb\)
            output = student\(torch\.cat\(\[rgb, motion_full\], dim=2\)\)
            prediction = output\[:, -1, 0\]
            saliency_loss = kd_saliency_loss\(prediction, targets\)
            prediction_low = prediction_low_sequence\.reshape\(-1, 3, LOW_H, LOW_W\)
            motion_loss, _ = motion_reconstruction_loss\(prediction_low, target_flow\)
            loss = saliency_loss \+ MOTION_REG_WEIGHT \* motion_loss''',
    flags=re.MULTILINE,
)

task_replacement = r'''        optimizer.zero_grad(set_to_none=True)

        # Privileged KD+SEA-RAFT output anchor.
        # The frozen student receives the cached original flow representation.
        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                kd_reference = student(inputs)[:, -1, 0].detach()

        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            motion_full, prediction_low_sequence = unified.infer_motion(rgb)
            output = student(
                torch.cat(
                    [rgb, motion_full],
                    dim=2,
                )
            )
            prediction = output[:, -1, 0]

            saliency_loss, saliency_parts = flowsal_task_loss(
                prediction,
                targets,
                kd_reference,
            )

            prediction_low = prediction_low_sequence.reshape(
                -1,
                3,
                LOW_H,
                LOW_W,
            )
            motion_loss, _ = motion_reconstruction_loss(
                prediction_low,
                target_flow,
            )

            loss = (
                saliency_loss
                + MOTION_REG_WEIGHT * motion_loss
            )'''

text, count = task_pattern.subn(
    task_replacement,
    text,
    count=1,
)

if count != 1:
    raise RuntimeError(
        "Could not patch the frozen-core task-training block."
    )

old_gate = '''    accuracy_gate = (
        task_deltas["dCC"] >= -0.010
        and task_deltas["dSIM"] >= -0.010
        and task_deltas["dKLD"] <= 0.050
    )'''

new_gate = '''    accuracy_gate = (
        task_deltas["dCC"] >= -0.002
        and task_deltas["dSIM"] >= -0.001
        and task_deltas["dKLD"] <= 0.005
        and task_deltas["dMSE"] <= 0.0003
    )'''

if old_gate not in text:
    raise RuntimeError(
        "Could not locate the original accuracy gate."
    )
text = text.replace(old_gate, new_gate)

old_thresholds = (
    '"accuracy_thresholds": '
    '{"dCC_min": -0.010, "dSIM_min": -0.010, '
    '"dKLD_max": 0.050}'
)

new_thresholds = (
    '"accuracy_thresholds": '
    '{"dCC_min": -0.002, "dSIM_min": -0.001, '
    '"dKLD_max": 0.005, "dMSE_max": 0.0003}'
)

if old_thresholds not in text:
    raise RuntimeError(
        "Could not locate the original threshold metadata."
    )
text = text.replace(old_thresholds, new_thresholds)

text = text.replace(
    "trained_unified_motion48_c8",
    "flowsal_v1_1",
)
text = text.replace(
    "trained_unified",
    "flowsal_v1_1",
)
text = text.replace(
    "Unified RGB motion-stem frozen-core probe",
    "FlowSal v1.1 balanced-anchor probe",
)
text = text.replace(
    "Unified ONNX:",
    "FlowSal ONNX:",
)

path.write_text(text, encoding="utf-8")
PY

    bash -n "$CHILD_SCRIPT"

    echo
    echo "===== PATCHED CHILD SCRIPT ====="
    sha256sum "$BASE_SCRIPT" "$CHILD_SCRIPT" \
        | tee "$STUDY_DIR/script_hashes.txt"

    if grep -q "kd_saliency_loss" "$CHILD_SCRIPT"; then
        echo "ERROR: The obsolete task loss remains in the child script."
        exit 1
    fi

    if ! grep -q "flowsal_task_loss" "$CHILD_SCRIPT"; then
        echo "ERROR: FlowSal task loss was not installed."
        exit 1
    fi

    RUN_MAP="$STUDY_DIR/run_map.tsv"
    printf \
        'seed\ttask_steps\trun_dir\treport\tmetrics\tonnx\tcheckpoint\n' \
        > "$RUN_MAP"

    read -r -a SEEDS <<< "$SEEDS_STRING"
    read -r -a TASK_STEPS_SET <<< "$TASK_STEPS_STRING"

    if [[ "${#SEEDS[@]}" -lt 3 ]]; then
        echo "ERROR: At least three seeds are required."
        exit 1
    fi

    for seed in "${SEEDS[@]}"; do
        for task_steps in "${TASK_STEPS_SET[@]}"; do
            echo
            echo "===================================================================================================="
            echo "FLOWSAL v1.1: SEED=$seed TASK_STEPS=$task_steps"
            echo "===================================================================================================="

            before_latest="$(
                find "$PROBE_ROOT/results" \
                    -mindepth 1 \
                    -maxdepth 1 \
                    -type d \
                    -printf '%T@ %p\n' \
                    2>/dev/null |
                sort -nr |
                head -n 1 |
                cut -d' ' -f2- ||
                true
            )"

            PAIR_EPOCHS="$PAIR_EPOCHS" \
            PAIR_BATCH="$PAIR_BATCH" \
            PAIR_LR="$PAIR_LR" \
            TASK_STEPS="$task_steps" \
            TASK_LR="$TASK_LR" \
            MOTION_REG_WEIGHT="$MOTION_REG_WEIGHT" \
            MAX_TRAIN_PAIRS="$MAX_TRAIN_PAIRS" \
            MAX_VAL_PAIRS="$MAX_VAL_PAIRS" \
            MAX_VAL_WINDOWS="$MAX_VAL_WINDOWS" \
            WARMUPS="$WARMUPS" \
            REPEATS="$REPEATS" \
            SEED="$seed" \
            bash "$CHILD_SCRIPT"

            after_latest="$(
                find "$PROBE_ROOT/results" \
                    -mindepth 1 \
                    -maxdepth 1 \
                    -type d \
                    -printf '%T@ %p\n' |
                sort -nr |
                head -n 1 |
                cut -d' ' -f2-
            )"

            if [[ -z "$after_latest" || ! -d "$after_latest" ]]; then
                echo "ERROR: No child result directory was created."
                exit 1
            fi

            if [[ "$after_latest" == "$before_latest" ]]; then
                echo "ERROR: The child run directory did not change."
                exit 1
            fi

            for required in \
                "$after_latest/report.json" \
                "$after_latest/saliency_summary.csv" \
                "$after_latest/per_sample_saliency_metrics.csv" \
                "$after_latest/checkpoints/unified_rgb_motion48_c8_probe.pt" \
                "$PROBE_ROOT/models/FlowSal_R192_T20_C8_v1_1_static_opset16.onnx"
            do
                if [[ ! -f "$required" ]]; then
                    echo "ERROR: Missing child artifact: $required"
                    exit 1
                fi
            done

            artifact_dir="$STUDY_DIR/artifacts/seed_${seed}/steps_${task_steps}"
            mkdir -p "$artifact_dir"

            cp -a \
                "$after_latest/report.json" \
                "$artifact_dir/report.json"

            cp -a \
                "$after_latest/saliency_summary.csv" \
                "$artifact_dir/saliency_summary.csv"

            cp -a \
                "$after_latest/per_sample_saliency_metrics.csv" \
                "$artifact_dir/per_sample_saliency_metrics.csv"

            cp -a \
                "$after_latest/checkpoints/unified_rgb_motion48_c8_probe.pt" \
                "$artifact_dir/flowsal_checkpoint.pt"

            cp -a \
                "$PROBE_ROOT/models/FlowSal_R192_T20_C8_v1_1_static_opset16.onnx" \
                "$artifact_dir/FlowSal_R192_T20_C8_v1_1_static_opset16.onnx"

            cp -a \
                "$after_latest/console.log" \
                "$artifact_dir/child_console.log"

            printf \
                '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$seed" \
                "$task_steps" \
                "$after_latest" \
                "$artifact_dir/report.json" \
                "$artifact_dir/per_sample_saliency_metrics.csv" \
                "$artifact_dir/FlowSal_R192_T20_C8_v1_1_static_opset16.onnx" \
                "$artifact_dir/flowsal_checkpoint.pt" \
                >> "$RUN_MAP"
        done
    done

    export FLOWSAL_STUDY_DIR="$STUDY_DIR"
    export FLOWSAL_MODEL_ROOT="$MODEL_ROOT"
    export FLOWSAL_RUN_MAP="$RUN_MAP"

    /home/mininet-ovs/venvs/satc/bin/python <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

study_dir = Path(os.environ["FLOWSAL_STUDY_DIR"])
model_root = Path(os.environ["FLOWSAL_MODEL_ROOT"])
run_map = Path(os.environ["FLOWSAL_RUN_MAP"])

thresholds = {
    "dCC_min": -0.002,
    "dSIM_min": -0.001,
    "dKLD_max": 0.005,
    "dMSE_max": 0.0003,
    "p95_max_ms": 1000.0 / 60.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


with run_map.open(
    "r",
    newline="",
    encoding="utf-8",
) as handle:
    mappings = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

if not mappings:
    raise RuntimeError("The FlowSal run map is empty.")

records = []

for mapping in mappings:
    report_path = Path(mapping["report"])
    report = json.loads(
        report_path.read_text(
            encoding="utf-8",
        )
    )

    summaries = report["validation"]["summaries"]
    baseline = summaries["original_raft"]
    candidate = summaries[
        "learned_motion_task_tuned"
    ]
    timing = report["timing"]["flowsal_v1_1"]

    deltas = {
        "dCC":
            float(candidate["CC"])
            - float(baseline["CC"]),
        "dSIM":
            float(candidate["SIM"])
            - float(baseline["SIM"]),
        "dKLD":
            float(candidate["KLD"])
            - float(baseline["KLD"]),
        "dMSE":
            float(candidate["MSE"])
            - float(baseline["MSE"]),
    }

    gate = (
        deltas["dCC"]
        >= thresholds["dCC_min"]
        and deltas["dSIM"]
        >= thresholds["dSIM_min"]
        and deltas["dKLD"]
        <= thresholds["dKLD_max"]
        and deltas["dMSE"]
        <= thresholds["dMSE_max"]
        and float(timing["p95_ms"])
        < thresholds["p95_max_ms"]
        and bool(timing["tensorrt_verified"])
    )

    records.append({
        "seed": int(mapping["seed"]),
        "task_steps": int(mapping["task_steps"]),
        "baseline_CC": float(baseline["CC"]),
        "baseline_SIM": float(baseline["SIM"]),
        "baseline_KLD": float(baseline["KLD"]),
        "baseline_MSE": float(baseline["MSE"]),
        "flowsal_CC": float(candidate["CC"]),
        "flowsal_SIM": float(candidate["SIM"]),
        "flowsal_KLD": float(candidate["KLD"]),
        "flowsal_MSE": float(candidate["MSE"]),
        **deltas,
        "mean_ms": float(timing["mean_ms"]),
        "p95_ms": float(timing["p95_ms"]),
        "p99_ms": float(timing["p99_ms"]),
        "calls_per_s": float(timing["calls_per_s"]),
        "tensorrt_verified":
            bool(timing["tensorrt_verified"]),
        "gate_pass": gate,
        "report": mapping["report"],
        "metrics": mapping["metrics"],
        "onnx": mapping["onnx"],
        "checkpoint": mapping["checkpoint"],
    })

candidate_csv = study_dir / "candidate_summary.csv"
fields = list(records[0].keys())

with candidate_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
    )
    writer.writeheader()
    writer.writerows(records)

by_steps = defaultdict(list)
for record in records:
    by_steps[record["task_steps"]].append(record)

step_rows = []

for steps in sorted(by_steps):
    items = by_steps[steps]

    output = {
        "task_steps": steps,
        "seeds": len(items),
        "passing_seeds":
            sum(
                int(item["gate_pass"])
                for item in items
            ),
    }

    for key in (
        "flowsal_CC",
        "flowsal_SIM",
        "flowsal_KLD",
        "flowsal_MSE",
        "dCC",
        "dSIM",
        "dKLD",
        "dMSE",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "calls_per_s",
    ):
        values = [
            float(item[key])
            for item in items
        ]
        output[f"{key}_mean"] = mean(values)
        output[f"{key}_std"] = (
            stdev(values)
            if len(values) > 1
            else 0.0
        )

    step_rows.append(output)

step_csv = study_dir / "step_aggregate_summary.csv"
with step_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(step_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(step_rows)

passing = [
    record
    for record in records
    if record["gate_pass"]
]

selected = None

if passing:
    # Predeclared rule:
    # among individually passing checkpoints,
    # choose highest validation CC.
    selected = max(
        passing,
        key=lambda item: (
            item["flowsal_CC"],
            item["flowsal_SIM"],
            -item["flowsal_KLD"],
            -item["flowsal_MSE"],
            -item["p95_ms"],
        ),
    )

    selected_onnx = (
        model_root
        / "FlowSal_R192_T20_C8_v1_1_Selected_static_opset16.onnx"
    )
    selected_checkpoint = (
        model_root
        / "FlowSal_R192_T20_C8_v1_1_Selected.pt"
    )

    shutil.copy2(
        selected["onnx"],
        selected_onnx,
    )
    shutil.copy2(
        selected["checkpoint"],
        selected_checkpoint,
    )

    selected["selected_onnx"] = str(selected_onnx)
    selected["selected_onnx_sha256"] = sha256(
        selected_onnx
    )
    selected["selected_checkpoint"] = str(
        selected_checkpoint
    )
    selected["selected_checkpoint_sha256"] = sha256(
        selected_checkpoint
    )

report = {
    "status": "success",
    "experiment":
        "FlowSal_v1_1_balanced_anchor_validation_study",
    "architecture":
        "FlowSal-R192-T20-C8",
    "runtime_input":
        "20 RGB frames at 144x192",
    "internal_motion_resolution":
        [36, 48],
    "motion_channels": 8,
    "added_motion_parameters": 811,
    "student_core_frozen": True,
    "sea_raft_executed": False,
    "cached_privileged_motion_used": True,
    "final_test_split_opened": False,
    "loss": {
        "ground_truth_weight": 0.50,
        "kd_raft_anchor_weight": 0.40,
        "teacher_map_weight": 0.10,
        "motion_regularization_weight": 0.05,
        "composite": {
            "KLD": 1.0,
            "CC_loss": 0.50,
            "SIM_loss": 0.25,
            "MSE": 10.0,
        },
    },
    "selection_thresholds": thresholds,
    "selection_rule":
        "highest validation CC among individually passing checkpoints",
    "candidate_count": len(records),
    "passing_candidate_count": len(passing),
    "selected": selected,
    "candidate_summary_csv": str(candidate_csv),
    "step_aggregate_summary_csv": str(step_csv),
}

report_path = study_dir / "report.json"
report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("FLOWSAL v1.1 CANDIDATE SUMMARY")
print("=" * 154)
print(
    f"{'Seed':>10}"
    f"{'Steps':>8}"
    f"{'CC':>11}"
    f"{'dCC':>11}"
    f"{'SIM':>11}"
    f"{'dSIM':>11}"
    f"{'KLD':>11}"
    f"{'dKLD':>11}"
    f"{'MSE':>11}"
    f"{'dMSE':>11}"
    f"{'P95 ms':>11}"
    f"{'Gate':>8}"
)
print("-" * 154)

for record in records:
    print(
        f"{record['seed']:>10d}"
        f"{record['task_steps']:>8d}"
        f"{record['flowsal_CC']:>11.6f}"
        f"{record['dCC']:>+11.6f}"
        f"{record['flowsal_SIM']:>11.6f}"
        f"{record['dSIM']:>+11.6f}"
        f"{record['flowsal_KLD']:>11.6f}"
        f"{record['dKLD']:>+11.6f}"
        f"{record['flowsal_MSE']:>11.6f}"
        f"{record['dMSE']:>+11.6f}"
        f"{record['p95_ms']:>11.3f}"
        f"{str(record['gate_pass']):>8}"
    )

print()
print("THREE-SEED AGGREGATE BY TASK STEPS")
print("=" * 142)
print(
    f"{'Steps':>8}"
    f"{'Pass':>8}"
    f"{'CC mean±std':>24}"
    f"{'dSIM mean±std':>24}"
    f"{'dKLD mean±std':>24}"
    f"{'dMSE mean±std':>24}"
    f"{'P95 mean±std':>24}"
)
print("-" * 142)

for row in step_rows:
    print(
        f"{row['task_steps']:>8d}"
        f"{row['passing_seeds']:>5d}/"
        f"{row['seeds']:<2d}"
        f"{row['flowsal_CC_mean']:>11.6f}"
        f"±{row['flowsal_CC_std']:<11.6f}"
        f"{row['dSIM_mean']:>+11.6f}"
        f"±{row['dSIM_std']:<11.6f}"
        f"{row['dKLD_mean']:>+11.6f}"
        f"±{row['dKLD_std']:<11.6f}"
        f"{row['dMSE_mean']:>+11.6f}"
        f"±{row['dMSE_std']:<11.6f}"
        f"{row['p95_ms_mean']:>11.3f}"
        f"±{row['p95_ms_std']:<11.3f}"
    )

print()
print("FLOWSAL v1.1 VALIDATION DECISION")
print("=" * 110)
print("Candidates evaluated:", len(records))
print("Passing candidates:", len(passing))
print("Final test split opened: False")

if selected is None:
    print("VALIDATION RESULT: NO-GO")
    print(
        "FlowSal v1 remains the frozen deployment baseline."
    )
else:
    print("VALIDATION RESULT: GO")
    print("Selected seed:", selected["seed"])
    print("Selected task steps:", selected["task_steps"])
    print(
        "Selected validation CC:",
        f"{selected['flowsal_CC']:.7f}",
    )
    print(
        "Selected validation deltas:",
        {
            key: selected[key]
            for key in (
                "dCC",
                "dSIM",
                "dKLD",
                "dMSE",
            )
        },
    )
    print(
        "Selected P95:",
        f"{selected['p95_ms']:.3f} ms",
    )
    print(
        "Selected ONNX:",
        selected["selected_onnx"],
    )
    print(
        "Selected ONNX SHA256:",
        selected["selected_onnx_sha256"],
    )

print("Report:", report_path)
print("Candidate CSV:", candidate_csv)
print("Step aggregate CSV:", step_csv)
PY

    echo
    echo "===== STUDY RESULT HASHES ====="
    (
        cd "$STUDY_DIR"
        find . \
            -type f \
            ! -name result_hashes.txt \
            -print0 |
        sort -z |
        xargs -0 sha256sum |
        tee result_hashes.txt
    )

    ln -sfn "$STUDY_DIR" "$EXP_ROOT/latest"

    echo
    echo "===== COMPLETE ====="
    echo "Study directory: $STUDY_DIR"
    echo "Report: $STUDY_DIR/report.json"
    echo "Candidate summary: $STUDY_DIR/candidate_summary.csv"
    echo "Step summary: $STUDY_DIR/step_aggregate_summary.csv"

    if [[ -f "$MODEL_ROOT/FlowSal_R192_T20_C8_v1_1_Selected_static_opset16.onnx" ]]; then
        echo "Selected FlowSal v1.1:"
        echo "$MODEL_ROOT/FlowSal_R192_T20_C8_v1_1_Selected_static_opset16.onnx"
    else
        echo "No candidate passed all predeclared gates."
        echo "FlowSal v1 remains unchanged."
    fi
} 2>&1 | tee "$CONSOLE_LOG"
