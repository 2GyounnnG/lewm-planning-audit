# Source verification for release v1.1

All 36,809 files covered by the R10 delivery seal were checked against the local delivery, including size and SHA256. The four preregistration identities and module-seal identities also match. Every newly copied release file matches its source bytes. No R3–R10 source file or seal was modified.

Original SHA256SUMS files are retained verbatim because they are historical source records. The R10 and R8-FIX source inventories include self-references whose recorded checksum cannot match the final inventory file. These are inventory defects, not changes to numerical result tables. New distribution checksums are generated separately under `manifests/checksums/` and exclude themselves.

The broader legacy-seal verification includes references to original remote training states, probe matrices and original H3X trajectory containers that are not present byte-for-byte in the local delivery. Numeric recovery artifacts do not establish byte identity with those original containers. These unavailable originals are explicitly recorded below; they are not counted as successful source-byte checks. This does not change the sealed study's existing recovery/degradation labels.

Verification counts: {"PASS_SOURCE_BYTES": 357842, "MATCHES_PREVIOUSLY_VERIFIED_ARCHIVE_INDEX": 43, "UNRESOLVED_SOURCE_PATH": 1021, "SOURCE_MANIFEST_SELF_REFERENCE": 3}.

The compact exception inventory is `SOURCE_VERIFICATION_EXCEPTIONS.json`. Detailed operation receipts are retained outside the public repository in the author's `publish_v1_1/` directory. No credential is included.
