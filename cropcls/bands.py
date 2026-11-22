"""Sentinel-2 band bookkeeping.

The SSL4EO-S12 pretrained weights in TorchGeo (``SENTINEL2_ALL_*``) expect the 13
Sentinel-2 L1C bands in this order. Every dataset in this package is reordered (or
sub-selected) to match it, and the first conv layer is sliced to the bands present.
"""

S2_ALL_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

# Sentinel-2 L2A (surface reflectance) products have no B10 (cirrus).
S2_L2A_BANDS: tuple[str, ...] = tuple(b for b in S2_ALL_BANDS if b != "B10")

S2_RGB_BANDS: tuple[str, ...] = ("B04", "B03", "B02")

BAND_PRESETS: dict[str, tuple[str, ...]] = {
    "all": S2_ALL_BANDS,
    "l2a": S2_L2A_BANDS,
    "rgb": S2_RGB_BANDS,
}


def normalize_band_name(name: str) -> str:
    """Accept 'B1', 'b01', 'B8a' ... and return the canonical 'B01' / 'B8A' form."""
    n = name.strip().upper()
    if not n.startswith("B"):
        raise ValueError(f"Not a Sentinel-2 band name: {name!r}")
    body = n[1:]
    if body == "8A":
        return "B8A"
    if body.isdigit():
        return f"B{int(body):02d}"
    raise ValueError(f"Not a Sentinel-2 band name: {name!r}")


def parse_bands(spec: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Parse a preset name ('all', 'l2a', 'rgb') or a comma separated list of bands."""
    if isinstance(spec, str):
        if spec.lower() in BAND_PRESETS:
            return BAND_PRESETS[spec.lower()]
        spec = [s for s in spec.split(",") if s.strip()]
    bands = tuple(normalize_band_name(b) for b in spec)
    unknown = [b for b in bands if b not in S2_ALL_BANDS]
    if unknown:
        raise ValueError(f"Unknown Sentinel-2 bands: {unknown}")
    if len(set(bands)) != len(bands):
        raise ValueError(f"Duplicate bands in {bands}")
    return bands
