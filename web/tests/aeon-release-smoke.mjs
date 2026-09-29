// Run against a separately relocated package launched with its offline wrapper.
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { chromium } from "@playwright/test";

const [baseUrl, outputPath] = process.argv.slice(2);
assert(
    baseUrl && outputPath,
    "Usage: node tests/aeon-release-smoke.mjs http://127.0.0.1:PORT output.json",
);
assert.equal(new URL(baseUrl).hostname, "127.0.0.1");
const browser = await chromium.launch({ headless: true });
const evidence = {
    base_url: baseUrl,
    status: "FAIL",
    checks: {},
    viewports: [],
    page_errors: [],
    external_requests: [],
};
try {
    const page = await browser.newPage();
    page.on("pageerror", (error) => evidence.page_errors.push(error.message));
    await page.route("**/*", (route) => {
        if (new URL(route.request().url()).origin !== new URL(baseUrl).origin) {
            evidence.external_requests.push(route.request().url());
            return route.abort();
        }
        return route.continue();
    });
    for (const endpoint of ["/health", "/api/v1/studies/aeon"]) {
        const response = await page.request.get(baseUrl + endpoint);
        assert.equal(response.status(), 200, endpoint);
        evidence.checks[endpoint] = "PASS";
    }
    const study = await (
        await page.request.get(baseUrl + "/api/v1/studies/aeon")
    ).json();
    const expected = {
        scale_direct: 1.178379078764592,
        scale_ema: 0.9134244980181899,
        expanded_direct: 0.640614632904252,
        expanded_ema: 0.6548551285493684,
    };
    assert.equal(
        study.scaling_development.slots.direct_seed7.final_pinball_db,
        expected.scale_direct,
    );
    assert.equal(
        study.scaling_development.slots.ema_jepa_seed7.final_pinball_db,
        expected.scale_ema,
    );
    assert.equal(
        study.expanded_train_development.slots.direct_seed7.final_pinball_db,
        expected.expanded_direct,
    );
    assert.equal(
        study.expanded_train_development.slots.ema_jepa_seed7.final_pinball_db,
        expected.expanded_ema,
    );
    assert.equal(study.expanded_train_development.joint_train_windows, 13472);
    evidence.checks.exact_reviewed_development_values = expected;
    const modelId = Object.keys(study.retrospective_test.models)[0];
    const replayResponse = await page.request.get(
        baseUrl +
            "/api/v1/studies/aeon/replay?model_id=" +
            encodeURIComponent(modelId),
    );
    assert.equal(replayResponse.status(), 200);
    const replay = await replayResponse.json();
    assert.equal(replay.total, 1216);
    assert.equal(replay.rows.length, 12);
    assert.equal(
        replay.source_time_basis,
        "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
    );
    assert(
        replay.rows.every((row) => !("truth" in row) && !("truth_db" in row)),
    );
    evidence.checks.historical_replay_total = replay.total;
    for (const width of [1440, 390]) {
        await page.setViewportSize({
            width,
            height: width === 390 ? 844 : 900,
        });
        await page.goto(baseUrl + "/#aeon-study");
        await page
            .getByRole("heading", {
                name: "Same-cohort 30k scaling development",
                exact: true,
            })
            .waitFor();
        await page
            .getByRole("heading", {
                name: "Expanded TRAIN 3k development",
                exact: true,
            })
            .waitFor();
        const scale = page.locator(
            'section[aria-labelledby="aeon-scaling-title"]',
        );
        const expanded = page.locator(
            'section[aria-labelledby="aeon-expanded-title"]',
        );
        assert((await scale.innerText()).includes("1.1784"));
        assert((await scale.innerText()).includes("0.9134"));
        assert((await expanded.innerText()).includes("0.6406"));
        assert((await expanded.innerText()).includes("0.6549"));
        assert((await expanded.innerText()).includes("13,472"));
        const geometry = await page.evaluate(() => ({
            viewport: innerWidth,
            document_scroll_width: document.documentElement.scrollWidth,
        }));
        evidence.viewports.push(geometry);
        assert(
            geometry.document_scroll_width <= geometry.viewport,
            JSON.stringify(geometry),
        );
        await page.screenshot({
            path: outputPath.replace(/\.json$/, `-${width}.png`),
            fullPage: true,
        });
    }
    assert.deepEqual(evidence.page_errors, []);
    assert.deepEqual(evidence.external_requests, []);
    evidence.status = "PASS";
} catch (error) {
    evidence.failure = String(error);
    throw error;
} finally {
    await browser.close();
    writeFileSync(outputPath, JSON.stringify(evidence, null, 2) + "\n");
    console.log(JSON.stringify(evidence));
}
