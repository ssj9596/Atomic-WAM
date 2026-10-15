"""Compact LAP-style language controls for DROID action windows.

The full LAP ``verbose_eef_with_rotation`` format includes translation
magnitudes and gripper state.  WAM subgoal conditioning only needs a compact
direction/rotation cue, so this module keeps the EEF-frame transformation and
direction vocabulary while omitting centimetres and gripper open/close.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _round_to_nearest_n(value: float, n: int) -> int:
    if n <= 0:
        raise ValueError(f"rotation precision must be positive, got {n}")
    return int(round(value / n) * n)


def _meets_threshold(value: float, threshold: float) -> bool:
    """Treat float32 values that represent the boundary as inclusive."""

    return value >= threshold or bool(np.isclose(value, threshold, rtol=1e-6, atol=1e-9))


def eef_window_action(
    cartesian_states: np.ndarray,
    interval: Sequence[int],
) -> np.ndarray:
    """Aggregate DROID Cartesian state changes in the initial EEF frame.

    This mirrors the validated LAP/DROID convention: translation is summed in
    the base frame and transformed by the initial rotation, while the y/z
    signs are flipped for the DROID EEF convention. Rotation uses the same
    conjugation and y/z Euler sign conversion as LAP.
    """

    states = np.asarray(cartesian_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 6:
        raise ValueError(f"cartesian_states must have shape [N, >=6], got {states.shape}")
    start, end = (int(interval[0]), int(interval[1]))
    if start < 0 or end <= start or end > states.shape[0]:
        raise ValueError(f"interval [{start}, {end}) is outside cartesian state range [0, {states.shape[0]})")

    translation = np.zeros(3, dtype=np.float64)
    rotation_total = np.eye(3, dtype=np.float64)
    for frame in range(start, end):
        if frame + 1 >= states.shape[0]:
            continue
        translation += states[frame + 1, :3] - states[frame, :3]
        current = Rotation.from_euler("xyz", states[frame, 3:6]).as_matrix()
        following = Rotation.from_euler("xyz", states[frame + 1, 3:6]).as_matrix()
        rotation_total = rotation_total @ (current.T @ following)

    initial = Rotation.from_euler("xyz", states[start, 3:6]).as_matrix()
    translation_eef = initial.T @ translation
    translation_eef[1] *= -1.0
    translation_eef[2] *= -1.0

    rotation_eef = initial.T @ rotation_total @ initial
    euler_eef = Rotation.from_matrix(rotation_eef).as_euler("xyz")
    euler_eef[1] *= -1.0
    euler_eef[2] *= -1.0
    return np.concatenate([translation_eef, euler_eef])


def format_lap_control(
    action: Sequence[float] | np.ndarray,
    *,
    include_translation: bool = True,
    include_rotation: bool = True,
    rotation_precision: int = 10,
    min_translation_cm: float = 2.0,
    min_rotation_degrees: float = 10.0,
) -> str:
    """Format an EEF action as qualitative control text.

    Translation keeps only direction (no centimetre magnitude). Rotation
    keeps direction and a rounded angle. Gripper state is intentionally absent.
    """

    if not np.isfinite(min_translation_cm) or min_translation_cm < 0:
        raise ValueError(f"min_translation_cm must be finite and non-negative, got {min_translation_cm}")
    if not np.isfinite(min_rotation_degrees) or min_rotation_degrees < 0:
        raise ValueError(f"min_rotation_degrees must be finite and non-negative, got {min_rotation_degrees}")

    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size < 6:
        raise ValueError(f"action must contain six pose values, got {values.size}")
    dx, dy, dz, droll, dpitch, dyaw = (float(value) for value in values[:6])
    translation_directions: list[str] = []
    if include_translation:
        if dx > 0 and _meets_threshold(abs(dx * 100.0), min_translation_cm):
            translation_directions.append("forward")
        elif dx < 0 and _meets_threshold(abs(dx * 100.0), min_translation_cm):
            translation_directions.append("backward")
        if dz > 0 and _meets_threshold(abs(dz * 100.0), min_translation_cm):
            translation_directions.append("upward")
        elif dz < 0 and _meets_threshold(abs(dz * 100.0), min_translation_cm):
            translation_directions.append("downward")
        if dy > 0 and _meets_threshold(abs(dy * 100.0), min_translation_cm):
            translation_directions.append("to the left")
        elif dy < 0 and _meets_threshold(abs(dy * 100.0), min_translation_cm):
            translation_directions.append("to the right")

    def join_phrases(parts: list[str]) -> str:
        if len(parts) < 2:
            return "".join(parts)
        if len(parts) == 2:
            return " and ".join(parts)
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"

    clauses: list[str] = []
    if translation_directions:
        clauses.append(f"guide the gripper {join_phrases(translation_directions)}")
    if include_rotation:
        droll_raw_deg = abs(droll * 180.0 / np.pi)
        dpitch_raw_deg = abs(dpitch * 180.0 / np.pi)
        dyaw_raw_deg = abs(dyaw * 180.0 / np.pi)
        droll_deg = (
            _round_to_nearest_n(droll_raw_deg, rotation_precision)
            if _meets_threshold(droll_raw_deg, min_rotation_degrees)
            else 0
        )
        dpitch_deg = (
            _round_to_nearest_n(dpitch_raw_deg, rotation_precision)
            if _meets_threshold(dpitch_raw_deg, min_rotation_degrees)
            else 0
        )
        dyaw_deg = (
            _round_to_nearest_n(dyaw_raw_deg, rotation_precision)
            if _meets_threshold(dyaw_raw_deg, min_rotation_degrees)
            else 0
        )
        subject = "it" if clauses else "the gripper"
        if droll > 0 and droll_deg:
            clauses.append(f"roll {subject} to the left around its forward axis by {droll_deg} degrees")
            subject = "it"
        elif droll < 0 and droll_deg:
            clauses.append(f"roll {subject} to the right around its forward axis by {droll_deg} degrees")
            subject = "it"
        if dpitch > 0 and dpitch_deg:
            clauses.append(f"tilt {subject} backward by {dpitch_deg} degrees")
            subject = "it"
        elif dpitch < 0 and dpitch_deg:
            clauses.append(f"tilt {subject} forward by {dpitch_deg} degrees")
            subject = "it"
        if dyaw > 0 and dyaw_deg:
            clauses.append(f"turn {subject} counterclockwise around its upward axis by {dyaw_deg} degrees")
        elif dyaw < 0 and dyaw_deg:
            clauses.append(f"turn {subject} clockwise around its upward axis by {dyaw_deg} degrees")
    if len(clauses) <= 1:
        return "".join(clauses)
    return f"{clauses[0]}, then {join_phrases(clauses[1:])}"


def format_rotation_control(
    action: Sequence[float] | np.ndarray,
    *,
    min_rotation_degrees: float,
    rotation_precision: int = 10,
) -> str:
    """Format only roll/pitch/yaw axes strictly above a degree threshold.

    Unlike :func:`format_lap_control`, this formatter never emits translation
    and joins simultaneous rotation axes as one coordinated action without a
    temporal ``then``.  The strict boundary is intentional: an axis at exactly
    ``min_rotation_degrees`` is omitted.
    """

    if not np.isfinite(min_rotation_degrees) or min_rotation_degrees < 0:
        raise ValueError(f"min_rotation_degrees must be finite and non-negative, got {min_rotation_degrees}")
    if rotation_precision <= 0:
        raise ValueError(f"rotation precision must be positive, got {rotation_precision}")

    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size < 6:
        raise ValueError(f"action must contain six pose values, got {values.size}")
    droll, dpitch, dyaw = (float(value) for value in values[3:6])

    clauses: list[str] = []
    subject = "the gripper"
    for radians, positive, negative in (
        (
            droll,
            "roll {subject} to the left around its forward axis",
            "roll {subject} to the right around its forward axis",
        ),
        (dpitch, "tilt {subject} backward", "tilt {subject} forward"),
        (
            dyaw,
            "turn {subject} counterclockwise around its upward axis",
            "turn {subject} clockwise around its upward axis",
        ),
    ):
        raw_degrees = abs(float(np.rad2deg(radians)))
        if raw_degrees <= min_rotation_degrees:
            continue
        rounded_degrees = _round_to_nearest_n(raw_degrees, rotation_precision)
        if rounded_degrees <= 0:
            continue
        template = positive if radians > 0 else negative
        clauses.append(f"{template.format(subject=subject)} by {rounded_degrees} degrees")
        subject = "it"

    if len(clauses) < 2:
        return "".join(clauses)
    if len(clauses) == 2:
        return " and ".join(clauses)
    return f"{', '.join(clauses[:-1])}, and {clauses[-1]}"


def append_rotation_purpose(subgoal: str, rotation_control: str) -> str:
    """Express a significant gripper rotation as the means to a subgoal."""

    base = " ".join(str(subgoal or "").split()).strip().rstrip(" .")
    cue = " ".join(str(rotation_control or "").split()).strip().rstrip(" .")
    if not cue:
        return base
    cue = cue[:1].upper() + cue[1:]
    if not base:
        return cue + "."

    # The ``lead`` object-position template starts with scene context, followed
    # by its executable action. Preserve that context as a complete sentence
    # and attach the rotation purpose to the action instead of producing the
    # ungrammatical ``... to the cup is behind ...``.
    context_prefix = ""
    first_sentence, separator, remainder = base.partition(". ")
    if separator and first_sentence.startswith("The ") and remainder:
        context_prefix = first_sentence.rstrip(" .") + ". "
        base = remainder.rstrip(" .")

    if len(base) > 1 and base[0].isupper() and base[1].islower():
        base = base[0].lower() + base[1:]
    return f"{context_prefix}{cue} to {base}."


def format_lap_control_for_interval(
    cartesian_states: np.ndarray,
    interval: Sequence[int],
    *,
    include_translation: bool = True,
    include_rotation: bool = True,
    rotation_precision: int = 10,
    min_translation_cm: float = 2.0,
    min_rotation_degrees: float = 10.0,
) -> str:
    return format_lap_control(
        eef_window_action(cartesian_states, interval),
        include_translation=include_translation,
        include_rotation=include_rotation,
        rotation_precision=rotation_precision,
        min_translation_cm=min_translation_cm,
        min_rotation_degrees=min_rotation_degrees,
    )


def append_lap_control(subgoal: str, control: str) -> str:
    """Prefix a stable, imperative control sentence before the gold subgoal."""

    base = " ".join(str(subgoal or "").split()).strip()
    cue = " ".join(str(control or "").split()).strip()
    if not cue:
        return base
    cue = cue[:1].upper() + cue[1:]
    if not base:
        return cue.rstrip(" .") + "."
    base = base.rstrip(" .")
    return f"{cue.rstrip(' .')}. Then {base[:1].lower() + base[1:]}."


__all__ = [
    "append_lap_control",
    "append_rotation_purpose",
    "eef_window_action",
    "format_lap_control",
    "format_lap_control_for_interval",
    "format_rotation_control",
]
