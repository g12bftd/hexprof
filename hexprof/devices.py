"""Target device constants.

Names and slugs are the ones Qualcomm AI Hub (Workbench) and AI Hub Models use,
so jobs, published numbers and GenieX model downloads line up.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetDevice:
    key: str
    hub_name: str  # AI Hub Workbench device name (hub.Device(name=...))
    chipset: str  # AI Hub chipset slug (geniex-bench --chipset, qai-hub-models -c)
    soc_model: str  # Android `ro.soc.model`
    description: str
    hexagon_arch: str
    # Published/independently measured facts (see README "Hardware facts").
    vtcm_bytes: int
    hvx_units: int
    hmx_units: int
    dram_peak_gbps: float  # LPDDR5X-5300 x 64-bit; the S25's real figure is undisclosed


GALAXY_S25 = TargetDevice(
    key="s25",
    hub_name="Samsung Galaxy S25",
    chipset="qualcomm-snapdragon-8-elite-for-galaxy",
    soc_model="SM8750",
    description="Snapdragon 8 Elite for Galaxy (SM8750-AC), Hexagon v79",
    hexagon_arch="v79",
    vtcm_bytes=8 * 1024 * 1024,
    hvx_units=6,
    hmx_units=1,
    dram_peak_gbps=84.8,
)

DEVICES = {d.key: d for d in [GALAXY_S25]}


def get_device(key_or_name: str) -> TargetDevice:
    """Look a device up by short key (``s25``) or AI Hub name."""
    if key_or_name in DEVICES:
        return DEVICES[key_or_name]
    for d in DEVICES.values():
        if d.hub_name.lower() == key_or_name.lower():
            return d
    # Unknown hosted device: keep the name, leave the hardware facts empty.
    return TargetDevice(
        key=key_or_name,
        hub_name=key_or_name,
        chipset="",
        soc_model="",
        description=key_or_name,
        hexagon_arch="",
        vtcm_bytes=0,
        hvx_units=0,
        hmx_units=0,
        dram_peak_gbps=0.0,
    )
