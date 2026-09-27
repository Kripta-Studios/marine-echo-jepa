import { describe, expect, it } from "vitest";
import { aeonContext } from "./aeonContext";
import type { AeonStudyEvidence } from "./api/client";

describe("AEON route context", () => {
  it("keeps development and unavailable packages free of TEST claims", () => {
    expect(aeonContext(null).modeLabel).toContain("No forecast replay");
    expect(
      aeonContext({
        retrospective_test_outcomes: "NOT_OPENED_FOR_THIS_REPORT",
      } as AeonStudyEvidence).uncertaintyLabel,
    ).toBe("No final evaluation");
  });

  it("labels a reviewed retrospective TEST replay without calling it sealed", () => {
    const context = aeonContext({
      retrospective_test_outcomes: "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST",
      retrospective_test: {
        status: "INDEPENDENTLY_REVIEWED_RETROSPECTIVE_TEST",
      },
    } as AeonStudyEvidence);
    expect(context.modeLabel).toBe("Reviewed retrospective replay");
    expect(context.uncertaintyLabel).toBe(
      "Retrospective evaluation · not sealed",
    );
  });
});
