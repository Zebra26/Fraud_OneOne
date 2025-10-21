#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from importlib.metadata import distributions, PackageNotFoundError

LICENSE_CANDIDATES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "COPYING.txt",
    "COPYING.md",
    "LICENCE",
    "LICENCE.txt",
    "LICENCE.md",
)


def extract_license(meta: dict[str, str]) -> tuple[str, list[str]]:
    # Prefer explicit License field
    lic = (meta.get("License") or meta.get("license") or "").strip()
    # Collect license classifiers
    classifiers = [c for c in meta.get("Classifier", []) if c.startswith("License ::")]
    return lic, classifiers


def normalize_meta_text(md) -> dict[str, object]:
    # importlib.metadata.Metadata is email.message.Message-like
    # Convert to simple dict with multi-keys expanded
    out: dict[str, object] = {}
    for k in md:
        vals = md.get_all(k)
        if vals is None:
            continue
        if len(vals) == 1:
            out[k] = vals[0]
        else:
            out[k] = list(vals)
    return out


GPL_PAT = re.compile(r"\bAGPL\b|\bGNU\s+Affero\b|\bGPL\b|General Public License", re.I)
LGPL_PAT = re.compile(r"Lesser General Public License", re.I)


def load_policy() -> dict:
    root = Path(__file__).resolve().parent.parent
    path = root / "license_policy.json"
    policy: dict = {
        "allow_packages": [],
        "deny_packages": [],
        "ignore_packages": [],
        "allow_licenses": ["LGPL"],
        "deny_licenses": ["AGPL", "GNU Affero", "GPL"],
        "treat_unknown_as_blocked": False,
    }
    try:
        if path.exists():
            user = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                policy.update(user)
    except Exception:
        pass

    def _compile(items):
        pats = []
        for it in items or []:
            try:
                pats.append(re.compile(it, re.I))
            except re.error:
                pats.append(re.compile(re.escape(str(it)), re.I))
        return pats

    policy["_allow_pkg"] = _compile(policy.get("allow_packages", []))
    policy["_deny_pkg"] = _compile(policy.get("deny_packages", []))
    policy["_ignore_pkg"] = _compile(policy.get("ignore_packages", []))
    policy["_allow_lic"] = _compile(policy.get("allow_licenses", []))
    policy["_deny_lic"] = _compile(policy.get("deny_licenses", []))
    return policy


def is_blocked_with_policy(name: str, license_name: str, classifiers: list[str], policy: dict) -> bool:
    def any_match(pats, s: str) -> bool:
        return any(p.search(s or "") for p in pats)

    # Package-level policy
    if any_match(policy.get("_ignore_pkg", []), name):
        return False
    if any_match(policy.get("_allow_pkg", []), name):
        return False
    if any_match(policy.get("_deny_pkg", []), name):
        return True

    # License-level policy (prefer classifiers)
    for c in classifiers or []:
        if any_match(policy.get("_deny_lic", []), c):
            return True
        if any_match(policy.get("_allow_lic", []), c):
            return False
    if license_name and len(license_name or "") < 64:
        if any_match(policy.get("_deny_lic", []), license_name):
            return True
        if any_match(policy.get("_allow_lic", []), license_name):
            return False

    # Heuristic GPL/AGPL block
    def flag(s: str) -> bool:
        return bool(GPL_PAT.search(s)) and not bool(LGPL_PAT.search(s))

    if classifiers:
        for c in classifiers:
            if flag(c):
                return True
        return False
    if license_name and len(license_name) < 64 and flag(license_name):
        return True
    if policy.get("treat_unknown_as_blocked") and not license_name and not classifiers:
        return True
    return False


def main() -> int:
    rows = []
    blocked = []
    license_blobs: list[tuple[str, str, str]] = []  # (name, version, text)
    policy = load_policy()
    for dist in sorted(distributions(), key=lambda d: d.metadata.get("Name", "").lower()):
        md = normalize_meta_text(dist.metadata)
        name = str(md.get("Name", ""))
        version = str(md.get("Version", ""))
        home = str(md.get("Home-page", md.get("Home-Page", "")))
        license_name, classifiers = extract_license(md)
        lic_classifiers = "; ".join(classifiers)
        blocked_flag = is_blocked_with_policy(name or "", license_name, classifiers, policy)
        if blocked_flag:
            blocked.append(name or f"<unknown>@{version}")
        rows.append(
            {
                "name": name,
                "version": version,
                "license": license_name or "UNKNOWN",
                "classifiers": classifiers,
                "home": home or "",
                "blocked": blocked_flag,
            }
        )

        # Try to locate a license file in the distribution contents
        try:
            files = list(dist.files or [])
        except Exception:
            files = []
        candidate_path = None
        for f in files:
            fn = f.name
            lower = fn.lower()
            if any(lower.endswith(c.lower()) or lower == c.lower() for c in LICENSE_CANDIDATES):
                candidate_path = f
                break
            if "license" in lower or "licence" in lower or "copying" in lower:
                candidate_path = f
                break
        if candidate_path is not None:
            try:
                path = dist.locate_file(candidate_path)
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
                if len(text) > 200_000:
                    text = text[:200_000] + "\n... (truncated)\n"
                license_blobs.append((name or "<unknown>", version or "", text))
            except Exception:
                pass

    # Write LICENSES.md
    out = Path(__file__).resolve().parent.parent / "LICENSES.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# Third-Party Licenses\n\n")
        f.write("This document lists third-party Python packages detected in the current environment and their licenses. Generated by `scripts/license_audit.py`.\n\n")
        if blocked:
            f.write("WARNING: The following packages appear to be GPL/AGPL-licensed (or include GPL in classifiers) and require legal review before use.\n\n")
            for b in blocked:
                f.write(f"- {b}\n")
            f.write("\n")
        # Table header
        f.write("| Package | Version | License | License Classifiers | Home |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                "| {name} | {version} | {license} | {classifiers} | {home} |\n".format(
                    name=r["name"],
                    version=r["version"],
                    license=(r["license"] or "UNKNOWN").replace("|", "/"),
                    classifiers=("; ".join(r["classifiers"]) if r["classifiers"] else "").replace("|", "/"),
                    home=(r["home"] or "").replace("|", "/"),
                )
            )

    # Also write machine-readable JSON
    out_json = Path(__file__).resolve().parent.parent / "third_party_licenses.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Wrote {out}")
    print(f"Wrote {out_json}")

    # Append license texts to LICENSES.md
    if license_blobs:
        with out.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write("## License Texts\n\n")
            for name, version, text in license_blobs:
                header = f"### {name} {version}".strip()
                f.write(header + "\n\n")
                f.write("```text\n")
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
                f.write("```\n\n")
    if blocked:
        print("BLOCKED: Potential GPL/AGPL detected:")
        for b in blocked:
            print(f" - {b}")
        # Non-zero exit to highlight issue in CI, but do not crash local dev unless desired.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
