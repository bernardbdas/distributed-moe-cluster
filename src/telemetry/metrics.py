"""Hardware FLOPS and Model FLOPs Utilization (MFU) Telemetry."""

from __future__ import annotations


class ModelMetricsTracker:
    """Calculates Model FLOPs Utilization (MFU) based on theoretical peak hardware capabilities."""

    # Peak theoretical TFLOPs (BF16 Tensor Core, Dense)
    HARDWARE_PEAK_TFLOPS = {
        "H100_SXM5": 989.0,
        "A100_SXM4": 312.0,
        "L40S": 366.0,
        "DEFAULT": 312.0,
    }

    def __init__(
        self,
        num_params: int,
        seq_len: int,
        hardware_type: str = "H100_SXM5",
    ):
        self.num_params = num_params
        self.seq_len = seq_len
        self.peak_tflops = self.HARDWARE_PEAK_TFLOPS.get(hardware_type, 312.0)

    def compute_mfu(self, tokens_processed: int, step_time_sec: float) -> float:
        """Calculates Model FLOPs Utilization (MFU).
        Standard transformer approximation: 6 * N * tokens per training step (forward + backward).
        """
        if step_time_sec <= 0:
            return 0.0

        flops_per_token = 6 * self.num_params
        total_flops = flops_per_token * tokens_processed
        achieved_tflops = (total_flops / step_time_sec) / 1e12

        mfu = achieved_tflops / self.peak_tflops
        return min(max(mfu, 0.0), 1.0)
