# Statements and Declarations

**Funding.** The author declares that no funds, grants, or other support were received during the preparation of this manuscript.

**Competing interests.** The author has no relevant financial or non-financial interests to disclose.

**Author contributions.** The author confirms sole responsibility for the study conception and design, data collection, analysis and interpretation of results, and manuscript preparation.

**Data availability.** The measurement artefacts are being prepared for public release in a versioned GitHub repository and Zenodo archive; until the DOI is assigned, the current package remains a pre-submission archive. The files include: the raw per-round latency logs for both latency protocols, the per-image accuracy outputs of the INT8 study together with their bootstrap resamples, the executed-graph node counts of Section 4.4 and Table S3, the frozen main-platform timing-image manifest, the SHA-256 mapping of the INT8 artifacts used for replication, and the scripts that generate every table and figure. The detectors themselves are the official COCO-pretrained checkpoints published by their respective projects, and the accuracy column used by LAE and by the Pareto analysis is the official COCO val2017 mAP published alongside those checkpoints rather than a value recomputed here (Section 3.4). The 500-image subset used for the FP32-to-INT8 comparison is a fixed, seeded sample of that validation set, and its own image list is archived with the same artefacts.

**Use of AI tools.** An AI assistant was used for language and editorial assistance and to cross-check that the numbers quoted in the prose match the archived measurement files. The study design, the measurements, the analysis, and the conclusions are the author's, and the author takes full responsibility for the content of the publication.
