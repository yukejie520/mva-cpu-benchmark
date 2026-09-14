from pathlib import Path
root=Path(__file__).resolve().parents[1]
repls={
"Whether a static INT8 model beats its FP32 baseline on this CPU is decided by the export format rather than by the quantization recipe.":"Whether a static INT8 model beats its FP32 baseline differs between the tested export paths and recipes; because activation types are not fully matched, this is a toolchain-specific observation rather than a format-only causal result.",
"In general we take rankings as the robust object of study (Section 4 shows ranking agreement across days and across a second CPU), and absolute cross-protocol numbers as range-bounded.":"In general we take within-platform rankings as the robust object of study, and absolute cross-protocol numbers as range-bounded.",
"Further rows add no step that these four do not already exercise, because the recommendation is always the feasible-set member with the highest LAE of Table 2.":"Further rows add no step that these four do not already exercise, because the recommendation follows the declared application priority within the feasible set.",
"Because the ranking that drives step 3 is exponent-robust (Section 4.3), \"recommend the highest LAE\" does not depend on a lucky choice of α and β.":"LAE sensitivity is reported separately and is not used to claim a superior selector.",
"The structural conclusions (FLOPs overstate CPU latency gaps, a joint budget query returns one recommendation, INT8 speed is decided by the export format rather than by the arithmetic) depend on direction and rank, which are the properties we verified.":"The structural conclusions (FLOPs overstate CPU latency gaps and explicit budgets yield an auditable lookup) depend on direction and rank. The INT8 contrast remains toolchain-specific.",
"**Data availability.** Every measurement artefact behind this paper is archived as a machine-readable file and is available from the corresponding author on reasonable request:":"**Data availability.** The measurement artefacts are being prepared for public release in a versioned GitHub repository and Zenodo archive; until the DOI is assigned, the current package remains a pre-submission archive. The files include:",
}
for p in [root/"paper/manuscript/3_methods.md",root/"paper/manuscript/4_results.md",root/"paper/manuscript/5_discussion.md",root/"paper/manuscript/8_declarations.md"]:
    s=p.read_text(encoding="utf-8")
    for a,b in repls.items(): s=s.replace(a,b)
    p.write_text(s,encoding="utf-8")
