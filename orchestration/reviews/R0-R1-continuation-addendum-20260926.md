# Independent continuation review addendum

Coordinator transcription of findings returned by the distinct read-only session
`/root/continuation_review` (configured `independent_reviewer`, GPT-5.6 Sol, high).
This is not coordinator self-approval. Historical findings in the original JSON remain
unchanged. No R0, R1, R2 or R3 blanket approval has been issued.

## Environmental helper and checkpoint preservation

The reviewer accepted the environmental helper at SHA-256
`a22cae1c9b4c2d9f385031fedfb323bbf2141b7512939c46a867a5caefbfbace`
and tests at `21bbd183a6bdde6af6719d2d850881cc238605b0181ca8fe90e5c979482d1deb`
as defensive parsing, matching and TEOS conversion code. It inspected the saved
19-test pass log; it did not execute pytest. This does not establish source QC,
fix-code semantics, environmental applicability or availability.

The numerical assay was accepted only as **nonpromotable numerical evidence**:
pinned Echopype/GSW, scalar discrepancy 2.84e-14 dB, frequency-specific finite ranges,
10–100 m common support and XML sound-speed sensitivity. The inspected report hash was
`2a4a324a` (reviewer supplied prefix); later report versions preserve earlier bytes.
No physically justified full-deployment calibration was established.

The reviewer independently invoked `Continue-Local.ps1 -Stage Campaign`, observed exit 2,
and verified that the existing registry hash remained
`6061823100962ded061123814acb8b20688c42f699036a590ca3afadd323fd69`.
It accepted the checkpoint-preservation fix, while confirming `executor_status=NOT_IMPLEMENTED`
and zero completed benchmarks. The wrapper's explicit final-test stop was accepted.

Verdicts remain R0 BLOCKED, R1 REQUEST_CHANGES, R2 NOT_ELIGIBLE, R3 NOT_REVIEWED.
The current compiled environmental profiles yield no candidate-test matches within
10 km/24 h. Required evidence is listed in the original review and ADR0004.

## Canonical store review at integrated commit 721c35a

Inspected store SHA-256:
`8dbb29450231fa3e24be9039b359ebe29b0a3543556b4e69209efcbe181ca401`.
Inspected preprocessing SHA-256:
`096ed666b7d81c59148ba998a6582e39ab81f088ad0cdabf4a7eb374ae4d0e8c`.
The reviewer did not execute pytest or open acoustic values.

The reviewer acknowledged fixes for test-target access, exact context/horizons/support,
full-band width/time weights, linear averaging, complete context/target containment,
configuration continuity, source/time disjointness, shard checksums and resume.
It requested these additional changes:

1. Separate integrity plumbing from independently approved physical calibration. Bind
   actual source inventory, configuration, calibration and R0 evidence in the real adapter.
2. Require contiguous UTC calendar splits and exact approved R1 protocol/split binding.
3. Reject unregistered/orphan shards during audit; a pruned manifest must not silently
   change eligibility denominators. Freeze the final processing manifest before promotion.
4. Separate measured availability from simulated zero-latency replay availability.
5. Durably record test-mask/metadata inspection before access, while distinguishing it
   from unopened acoustic values.

The canonical schema also lacks some full data-contract fields (dataset version,
calibration status, QC reasons, position/time and orientation), which must remain explicit
before a real corpus adapter exists. Synthetic tests do not establish physical provenance.

The reviewer clarified the acceptable boundary: a canonical store may establish byte
integrity and deterministic lineage and remain explicitly
`NONPROMOTABLE_ENGINEERING_FIXTURE`. A separate real-corpus adapter must verify preserved
independent R0/R1 records bound to the exact source/config/calibration/protocol artifacts.
File hashes establish integrity; the distinct session history establishes governance.
Current BLOCKED calibration evidence must never satisfy that gate. Builder fixes are pending
renewed review; this addendum does not pre-approve them.

## Deployed serial and certificate provenance

The independent reviewer subsequently checked source metadata online and parsed only
metadata from the predetermined training file. PANGAEA949811, both XML/DPL configurations,
the raw metadata, and [O2A parent4304](https://registry.o2a-data.de/rest/v2/items/4304?with=resources)
support **deployed downward-looking AZFP55170**. The manual's November2019 down55169
entry is stale predeployment documentation and does not outweigh those records.

The [linked sensor item5748](https://registry.o2a-data.de/rest/v2/items/5748?with=resources)
is internally inconsistent: its code/name say55170, but its serial field says55171
and attached resource1246 is a calibration certificate for55171. It cannot certify55170.
The remaining finding is **missing authoritative calibration provenance for deployed55170**,
not an unresolved choice of deployed serial. XML coefficients remain unverified physical
calibration inputs. R0 remains BLOCKED.

The source manual must stay preserved locally, but its operational credential-like strings
must not be copied into release payloads or evidence excerpts. No such strings are recorded here.

## Certificate correction following content inspection

The independent reviewer followed the corresponding upward-instrument registry record and
found that the two resources are swapped. The public [resource980 PDF](https://registry.o2a-data.de/rest/v2/items/4308/resources/980/payload)
actually names serial55170. It is the ASL factory calibration certificate v12 dated
26 March2019. Download size207370 bytes, publisher MD5
`d064edfe9088bfd4260406836e88f159`, SHA-256
`6541fbe21f6af64e0731f53784a2763b87d8527ff79ddcba178bd0dd5eb3f464`.
The coordinator independently downloaded and verified those bytes; provenance is in
`evidence/continuation/factory_certificate.json`.

The reviewer compared its four frequency TVR/VTX/BP/EL/DS coefficients with the deployment
XML and found agreement to the certificate's printed precision. Certificate sphere-check
errors are -0.8, -0.5, 0 and +0.7 dB for38/125/200/455kHz. Its narrow disposition is
**APPROVE serial55170 factory coefficient mapping and record provenance**. This resolves
the missing certificate finding above; it does not establish field calibration or
environmental/sound-path/QC validity. Earlier findings remain historical evidence.

## Range integration primitive

The reviewer accepted `range_grid.py` at
`21ef36fa66d4aa762fe1aa85e8a39d03a0a6e61531514d2aa5ae1aef8a4f5b33`
as a narrow numerical engineering primitive. It inspected the seven-test green log but
did not execute pytest. Acceptance covers overlap integration of linear sv on explicitly
supplied edges, full-width denominators and conservative masks, not edge derivation,
physical range regridding justification or real-corpus calibration.

## Trajectory coverage correction

The coordinator found that the first numerical assay used Iridium-only positions for
environmental matching despite long Iridium gaps. XEOS records contain additional validly
parsed position timestamps. `trajectory_matches.json` reports all three policies rather
than erasing the old result. Nearest-time selection across both preserved trackers yields
99/100 train,25/25 validation,16/16 calibration and12/24 test profile-day matches at the
unchanged illustrative5km/24h screen. All four newly downloaded recovered SIT buoys have
zero such matches. These are metadata screens, not physical calibration approvals.

The distinct reviewer accepted the corrected trajectory audit narrowly as descriptive metadata
evidence. It does not approve the illustrative matching thresholds as physical eligibility.

## Fixed regional sensitivity and follow-up

The reviewer independently inspected the preregistered assay result and returned
REQUEST_CHANGES/BLOCKED for physical promotion: the primary conditional sound-path bound
is 1.080721198 dB, above the unchanged 1 dB limit. Range displacement 1.688427 m is below
2 m; homogeneous 0.905364 dB cannot override the failed path bound. Upstream/scalar maximum
disagreement is 6.21e-14 dB. Other frequency bounds are 2.391/2.811/5.614 dB.

The reviewer endorsed a prospective depth-stratified follow-up while preserving the failed
assay, original sources/QC/allowances and thresholds. The actual-specific public installation
record supports a downward instrument on a 5 m umbilical, not exact time-varying depth or tilt.
No measured ice draft, cable angle or absolute transducer-face depth was found. Depth scenarios
therefore remain engineering assumptions and cannot by themselves establish field calibration.

The depth-assay first source execution failed on upstream scalar-salinity branching; its
traceback is retained. The reviewer checked the subsequent vectorized equation against pinned
Echopype 0.11.1 and approved the numerical transcription narrowly. The four-test GREEN log
includes scalar upstream comparisons; the reviewer did not execute pytest.

## Fixture store, baseline executor and evidence reporting

The reviewer approved the final fixture store after exposure-log hardening, including durable
pre-access recording of protected metadata/masks. Synthetic fixture logs are not real holdout
exposure. Neither the store nor its adapter constitutes an R0/R1-approved real-corpus path.

Initial baseline review requested fixes to all-partition source separation, exact frequency/
range grids, per-shard source binding and resume metric/content validation. These were resolved
in builder commits 4155e9c/f3d5c7e, integrated as ca6ccef/33d5d76. The reviewer approved only
the synthetic-fixture engineering scope after inspecting saved 11-test evidence; the coordinator
then ran the integrated baseline/report tests (18 passed). No benchmark execution was approved.

The reviewer also approved the evidence-preserving CLI report correction: existing registry
attempts and exposure are returned intact; FAILED/RUNNING/completed subsets report INCOMPLETE,
not NOT_RUN. The saved final focused log has seven passing tests. This is reporting approval,
not completion of the campaign or any scientific gate.
