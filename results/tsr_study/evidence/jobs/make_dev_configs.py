"""Write the development configs allowed by docs/TSR_STUDY.md: smoke seeds/sizes, and a timing grid on seed 99999."""
import sys, tomllib
from pathlib import Path
code = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(code / "scripts"))
from make_reference_configs import write_config
study = tomllib.loads((code / "configs/tsr_study.toml").read_text())
smoke = tomllib.loads((code / "configs/gamma_smoke.toml").read_text())
run = {**smoke["run"], "methods": study["run"]["methods"], "reference_method": study["run"]["reference_method"],
       "scope": "TSR development smoke on the gamma generator (outside the catalog); execution status only", "purpose": "smoke"}
write_config({"schema_version": 1, "run": run, "method_options": study["method_options"], "scenarios": smoke["scenarios"]}, out / "smoke.toml")
timing = [s for s in study["scenarios"] if s["label"] == "o2048_r512"]
run = {**study["run"], "seeds": [99999], "purpose": "smoke", "scope": "TSR timing preflight on seed 99999 (outside the study grid); runtime and status only"}
write_config({"schema_version": 1, "run": run, "method_options": study["method_options"], "scenarios": timing}, out / "timing.toml")
print("smoke scenarios", len(smoke["scenarios"]), "seeds", smoke["run"]["seeds"], "| timing scenarios", len(timing))
