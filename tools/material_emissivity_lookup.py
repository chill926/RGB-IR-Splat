"""Offline emissivity candidate lookup and material-config template generator.

The table is a starting point, not an automatic measurement. Values depend on
surface finish, oxidation, wavelength, temperature and view angle. Generated
entries are intentionally ``confirmed: false``; inspect the scene and source,
then choose a value and mark each entry true before training.
"""
import argparse
import json
from pathlib import Path


SOURCE_FLIR_TABLE = "https://support.flir.com/DSDownload/Assets/T810442-en-US_A4.pdf"
SOURCE_FLIR_GUIDE = "https://www.flir.com/discover/professional-tools/how-does-emissivity-affect-thermal-imaging/"

CANDIDATES = {
    "paint_flat": {
        "range": [0.92, 0.94], "suggested": 0.93, "spectrum": "LWIR table range",
        "surface": "paint; eight colors/qualities in the cited LWIR table", "source": SOURCE_FLIR_TABLE,
    },
    "rubber": {
        "range": [0.93, 0.97], "suggested": 0.95, "spectrum": "broadband table value",
        "surface": "hard or rough rubber near room temperature", "source": SOURCE_FLIR_TABLE,
    },
    "glass_float_uncoated": {
        "range": [0.94, 0.97], "suggested": 0.97, "spectrum": "LWIR table value",
        "surface": "uncoated float-glass surface", "source": SOURCE_FLIR_TABLE,
    },
    "steel_polished": {
        "range": [0.05, 0.28], "suggested": 0.14, "spectrum": "condition dependent",
        "surface": "clean/polished steel; strongly reflective", "source": SOURCE_FLIR_TABLE,
    },
    "steel_oxidized": {
        "range": [0.61, 0.85], "suggested": 0.74, "spectrum": "condition dependent",
        "surface": "oxidized or rusty steel", "source": SOURCE_FLIR_TABLE,
    },
    "asphalt_paving": {
        "range": [0.95, 0.98], "suggested": 0.967, "spectrum": "LLW table value",
        "surface": "asphalt paving near room temperature", "source": SOURCE_FLIR_TABLE,
    },
    "concrete": {
        "range": [0.92, 0.974], "suggested": 0.95, "spectrum": "condition-dependent table values",
        "surface": "ordinary/dry/rough concrete", "source": SOURCE_FLIR_TABLE,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--query", choices=sorted(CANDIDATES))
    parser.add_argument("--template", nargs="+", choices=sorted(CANDIDATES),
                        help="Candidate keys to include in a JSON material template")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CANDIDATES, indent=2, ensure_ascii=False))
        return
    if args.query:
        print(json.dumps({args.query: CANDIDATES[args.query]}, indent=2, ensure_ascii=False))
        return
    if not args.template or not args.output:
        parser.error("Use --list, --query KEY, or --template KEY... --output FILE")
    materials = []
    for material_id, key in enumerate(args.template):
        candidate = CANDIDATES[key]
        materials.append({
            "id": material_id,
            "name": key,
            "epsilon0": candidate["suggested"],
            "epsilon0_candidate_range": candidate["range"],
            "surface_condition": candidate["surface"],
            "spectrum": candidate["spectrum"],
            "source": candidate["source"],
            "confirmed": False,
            "learn_k": key not in {"glass_float_uncoated", "steel_polished"},
            "learn_delta": key not in {"glass_float_uncoated", "steel_polished"},
            "k_prior": 0.0,
            "sigma_k": 0.0001,
            "sigma_delta": 0.05,
        })
    document = {
        "format_version": 1,
        "spectral_band": "8-14 um approximation; verify against the actual camera response",
        "temperature_reference_K": 300.0,
        "unknown_label": 255,
        "unknown_epsilon0": 0.95,
        "materials": materials,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
    print(f"Wrote unconfirmed template: {output}")


if __name__ == "__main__":
    main()
