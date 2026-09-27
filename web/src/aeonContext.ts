import type { AeonStudyEvidence } from "./api/client";

export function aeonContext(study: AeonStudyEvidence | null) {
  const reviewedTest =
    study?.retrospective_test_outcomes ===
      "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST" &&
    study.retrospective_test?.status ===
      "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST";
  return {
    cutoffLabel: "Source clock unspecified",
    modeLabel: reviewedTest
      ? "Reviewed retrospective replay"
      : "Validation evidence · No forecast replay",
    uncertaintyLabel: reviewedTest
      ? "Retrospective evaluation · not sealed"
      : "No final evaluation",
  };
}
