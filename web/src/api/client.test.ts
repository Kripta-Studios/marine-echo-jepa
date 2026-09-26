import { describe, expect, it } from "vitest";
import { forecastRequest } from "./client";

describe("forecast request boundary", () => {
  it("sends only identifiers and cutoff, never revealed observations", () => {
    const state = {
      datasetId: "real",
      modelId: "direct",
      cutoff: "2020-02-17T12:00:00Z",
      age: 0,
      futureTruth: [100],
    };
    expect(forecastRequest(state)).toEqual({
      dataset_id: "real",
      model_id: "direct",
      cutoff: state.cutoff,
      observation_age_hours: 0,
      mode: "cached_replay",
    });
  });
});
