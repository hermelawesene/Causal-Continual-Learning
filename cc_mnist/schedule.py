"""
schedule.py
===========
Implements the per-epoch SCHEDULE = {alpha_t, p_t, clarity_on, lambda_diff_t, T_t}
from doc1, organized around the user's requested three-phase pipeline:

    Phase 1 "warmup"      epochs [0, warmup_epochs)                         : alpha=0   (no gating at all)
    Phase 2 "soft gating" epochs [warmup_epochs, warmup_epochs+soft_epochs) : alpha ramps 0 -> alpha_start..mid
    Phase 3 "full CC"     remaining epochs                                  : alpha ramps to alpha_end,
                                                                               clarity turns on and ramps too

This is a deliberate (and, we think, sensible) elaboration of doc1's own
"Suggested schedules" section, which only specifies the *shape* of the
ramps (e.g. "alpha_t: 0.05 -> 0.60 over epochs") without a phase
structure. See README "On the phased training pipeline" for the
reasoning and for an alternative, loss-plateau-triggered phase
transition we also implement as an option.
"""
from dataclasses import dataclass


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


class CCSchedule:
    def __init__(self, cfg, total_epochs: int = None):
        self.cfg = cfg
        self.total_epochs = total_epochs or cfg.total_epochs
        self.phase1_end = cfg.warmup_epochs
        self.phase2_end = cfg.warmup_epochs + cfg.soft_gate_epochs
        self.phase3_len = max(1, self.total_epochs - self.phase2_end)

    def phase_name(self, epoch: int) -> str:
        if epoch < self.phase1_end:
            return "warmup"
        if epoch < self.phase2_end:
            return "soft_gate"
        return "full_cc"

    def get(self, epoch: int) -> dict:
        cfg = self.cfg
        phase = self.phase_name(epoch)

        if phase == "warmup":
            alpha = 0.0
            p = cfg.p_start
            clarity_on = False
            lambda_diff = 0.0
        elif phase == "soft_gate":
            t = (epoch - self.phase1_end) / max(1, cfg.soft_gate_epochs)
            # ramp alpha from 0 up to a "soft" midpoint, well below alpha_end
            alpha = _lerp(0.0, cfg.alpha_start, t)
            p = _lerp(cfg.p_start, (cfg.p_start + cfg.p_end) / 2.0, t)
            clarity_on = False
            lambda_diff = 0.0
        else:
            t = (epoch - self.phase2_end) / self.phase3_len
            alpha = _lerp(cfg.alpha_start, cfg.alpha_end, t)
            p = _lerp((cfg.p_start + cfg.p_end) / 2.0, cfg.p_end, t)
            epochs_into_cc = epoch - self.phase2_end
            clarity_on = epochs_into_cc >= cfg.clarity_warmup_epochs
            if clarity_on:
                t_clarity = (epochs_into_cc - cfg.clarity_warmup_epochs) / max(1, self.phase3_len - cfg.clarity_warmup_epochs)
                lambda_diff = _lerp(cfg.lambda_diff_start, cfg.lambda_diff_end, t_clarity)
            else:
                lambda_diff = 0.0

        temperature = _lerp(cfg.temperature_start, cfg.temperature_end, epoch / max(1, self.total_epochs - 1))

        return dict(
            phase=phase,
            alpha_t=alpha,
            p_t=p,
            clarity_on=clarity_on,
            lambda_diff_t=lambda_diff,
            T_t=temperature,
            diffusion_order=cfg.diffusion_order,
            gate_floor_min_fraction=cfg.gate_floor_min_fraction,
            clarity_every_n_batches=cfg.clarity_every_n_batches,
        )
