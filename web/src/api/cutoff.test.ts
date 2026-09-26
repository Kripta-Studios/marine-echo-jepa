import { expect, test } from "vitest";
import { safeCutoff } from "./cutoff";

test("invalid and unsupported URL cutoffs recover to the diagnostic default", () => {
  for (const value of [
    null,
    "garbage",
    "2020-02-18T01:00:00Z",
    "2020-02-17T12:00:00",
  ]) {
    expect(safeCutoff(value)).toBe("2020-02-17T12:00:00Z");
  }
  expect(safeCutoff("2020-02-17T18:00:00Z")).toBe("2020-02-17T18:00:00Z");
});
