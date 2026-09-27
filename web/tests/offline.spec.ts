import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { writeFileSync } from 'node:fs';

test('offline real replay, unavailable forecast, reveal, evidence and export', async ({page}) => {
  const external: string[] = [];
  const errors: string[] = [];
  const forecasts: object[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.hostname !== '127.0.0.1') { external.push(url.href); return route.abort(); }
    if (url.pathname === '/api/v1/forecast') forecasts.push(route.request().postDataJSON());
    return route.continue();
  });
  await page.setViewportSize({width:1440,height:900});
  const before = page.waitForResponse(response => response.url().includes('/observations?'));
  await page.goto('/');
  const initial = await (await before).json();
  expect(initial.data_kind).toBe('public_real');
  expect(initial.calibrated).toBe(false);
  expect(initial.rows.length).toBe(48);
  expect(initial.rows.every((row: {event_time_utc: string}) => Date.parse(row.event_time_utc) <= Date.parse('2020-02-17T12:00:00Z'))).toBe(true);
  await expect(page.getByRole('img', {name:/Observed before cutoff/})).toBeVisible();
  await page.screenshot({path:resolve(import.meta.dirname, '../../evidence/browser/desktop.png'),fullPage:true});
  await page.getByRole('button', {name:/Forecast here/}).click();
  await expect(page.getByRole('heading', {name:'Forecast unavailable'})).toBeVisible();
  await page.getByRole('button', {name:'Reveal observed outcome'}).click();
  await expect(page.getByRole('img', {name:/Revealed later observations/})).toBeVisible();
  await page.getByRole('button', {name:/Forecast here/}).click();
  await expect.poll(() => forecasts.length).toBe(2);
  expect(forecasts[0]).toEqual(forecasts[1]);
  expect(Object.keys(forecasts[0])).toEqual(['dataset_id','model_id','cutoff','observation_age_hours','mode']);
  await page.getByRole('link', {name:/Forecast comparison/}).click();
  await expect(page.getByRole('heading', {name:'Compare forecasts'})).toBeVisible();
  await page.getByRole('link', {name:/Observation freshness/}).click();
  await expect(page.getByText('Predefined age mask · replay simulation')).toBeVisible();
  await page.getByRole('link', {name:/Experiment lab/}).click();
  const recordedModels = await (await page.request.get('/api/v1/models')).json();
  expect(recordedModels.length).toBeGreaterThan(0);
  expect(recordedModels.every((model: {status: string}) => model.status !== 'AVAILABLE')).toBe(true);
  await expect(page.getByRole('cell', {name:recordedModels[0].status, exact:true}).first()).toBeVisible();
  await expect(page.getByText('Required run registry (25 runs)')).toBeVisible();
  await page.getByRole('link', {name:/Evidence & transfer/}).click();
  const health = await (await page.request.get('/health')).json();
  await expect(page.getByText(health.release_class, {exact:true})).toBeVisible();
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', {name:/Export JSON/}).click();
  const file = await downloaded;
  expect(file.suggestedFilename()).toBe('marine-echo-evidence.json');
  await page.screenshot({path:resolve(import.meta.dirname, '../../evidence/browser/evidence.png'),fullPage:true});
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});

test('malformed cutoff recovers and every sample group is accessible', async ({page}) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?cutoff=not-a-date');
  await expect(page.getByLabel('Prediction cutoff (UTC)')).toHaveValue('2020-02-17T12:00:00Z');
  await page.getByText('Accessible observation table (48 bins)').click();
  await expect(page.getByRole('columnheader', {name:'Group 63 · raw counts', exact:true})).toBeAttached();
  expect(errors).toEqual([]);
});

test('changing cutoff cancels pending reveal without leaking stale data or errors', async ({page}) => {
  await page.goto('/');
  await expect(page.getByRole('img', {name:/Observed before cutoff/})).toBeVisible();
  await page.route('**/observations?*', async route => {
    if (!new URL(route.request().url()).searchParams.has('cutoff')) {
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    await route.continue().catch(() => {});
  });
  await page.getByRole('button', {name:'Reveal observed outcome'}).click();
  await page.getByLabel('Prediction cutoff (UTC)').selectOption('2020-02-17T18:00:00Z');
  await expect(page.getByRole('img', {name:/Observed before cutoff/})).toBeVisible();
  await page.waitForTimeout(650);
  await expect(page.getByRole('img', {name:/Revealed later observations/})).toHaveCount(0);
  await expect(page.getByText(/aborted|AbortError/i)).toHaveCount(0);
});

test('mobile fits, keyboard focus and refresh preserve cutoff', async ({page}) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/?cutoff=2020-02-17T18%3A00%3A00Z');
  await expect(page.getByRole('img', {name:/Observed before cutoff/})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.keyboard.press('Tab');
  expect(await page.evaluate(() => document.activeElement?.tagName)).toBe('A');
  await page.reload();
  await expect(page.getByLabel('Prediction cutoff (UTC)')).toHaveValue('2020-02-17T18:00:00Z');
  await page.screenshot({path:resolve(import.meta.dirname, '../../evidence/browser/mobile.png'),fullPage:true});
});

test('bounded replay navigation meets the declared latency target', async ({page}) => {
  await page.goto('/');
  await expect(page.getByRole('img', {name:/Observed before cutoff/})).toBeVisible();
  const samples: number[] = [];
  for (let i = 0; i < 20; i++) {
    const hour = i % 2 ? 18 : 6;
    const started = performance.now();
    await page.getByLabel('Prediction cutoff (UTC)').selectOption(`2020-02-17T${String(hour).padStart(2,'0')}:00:00Z`);
    await expect(page.getByRole('img', {name:`Observed before cutoff: ${hour*4} observed time bins; raw counts, sample-index axis. Accessible table below.`,exact:true})).toBeVisible();
    samples.push(performance.now()-started);
  }
  const sorted = [...samples].sort((a,b) => a-b);
  const p95 = sorted[Math.ceil(.95*sorted.length)-1];
  writeFileSync(resolve(import.meta.dirname,'../../evidence/browser/navigation-latency.json'),JSON.stringify({metric:'UI cutoff change through rendered observation canvas',samples_ms:samples,p95_ms:p95,target_ms:300,live_forecast_latency:null,live_forecast_status:'NOT_RUN_NO_MODEL'},null,2));
  expect(p95).toBeLessThanOrEqual(300);
});

test('raw TRAIN-development evidence shows all predictions and current release class', async ({page}) => {
  const cutoffs = [
    '2020-03-01T00:00:00Z',
    '2020-03-02T00:00:00Z',
  ];
  const horizons = [1, 3, 6].map((horizon_hours) => ({
    horizon_hours,
    eligible_rows: 2,
    target_days: 2,
    ridge: { daily_mean_pinball_code: 1, mae_code_median: 2 },
    direct_neural: { daily_mean_pinball_code: 3, mae_code_median: 4 },
  }));
  const artifact = {
    study_id: 'raw_response_development_v1',
    run_id: 'browser-contract-fixture',
    status: 'REAL_TRAIN_DEVELOPMENT_ONLY',
    quantity: 'complete_positive_azfp_backscatter_r_code_mean',
    unit: 'transformed AZFP response code (not calibrated Sv)',
    calibrated: false,
    final_evaluation: false,
    comparison_label: 'Direct neural is worse than ridge in this contract fixture.',
    direct_checkpoint_128_sha256: 'fixture-checkpoint-not-real',
    horizons,
    rows: cutoffs.map((cutoff_utc, cutoff_index) => ({
      cutoff_utc,
      horizons: [1, 3, 6].map((horizon_hours) => ({
        horizon_hours,
        target_start_utc: `2020-03-0${cutoff_index + 2}T${String(horizon_hours).padStart(2, '0')}:00:00Z`,
        eligible: true,
        truth_code: 10 + cutoff_index + horizon_hours,
        ridge_quantiles_code: [1, 2, 3, 4, 5],
        direct_quantiles_code: [6, 7, 8, 9, 10],
      })),
    })),
    limitations: ['Browser contract fixture only; no experimental claim.'],
  };
  await page.route('**/api/v1/evidence/raw-development', (route) =>
    route.fulfill({ status: 200, json: artifact }),
  );
  await page.route('**/api/v1/evidence/research', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({ response, json: { ...body, release_class: 'OFFLINE_RESEARCH_ENGINEERING_ONLY' } });
  });

  await page.goto('/#experiment-lab');
  await expect(page.getByRole('heading', { name: 'Raw response-code TRAIN development' })).toBeVisible();
  const resultTable = page.getByRole('table', { name: /All TRAIN-development predictions and truths/ });
  await expect(resultTable.locator('tbody tr')).toHaveCount(6);
  await expect(resultTable.getByRole('columnheader', { name: 'Ridge codes (5%, 25%, 50%, 75%, 95%)' })).toBeAttached();
  await expect(resultTable.getByRole('cell', { name: '12', exact: true })).toBeVisible();
  await expect(resultTable.getByRole('cell', { name: '17', exact: true })).toBeVisible();
  await expect(page.getByText('Direct neural is worse than ridge on both reported metrics at all three horizons.')).toBeVisible();

  await page.getByRole('link', { name: /Evidence & transfer/ }).click();
  await expect(page.getByText('OFFLINE_RESEARCH_ENGINEERING_ONLY', { exact: true })).toBeVisible();
  await expect(page.getByText(/calibrated core and final evaluation remain blocked/i)).toBeVisible();
});

test('missing raw TRAIN-development artifact has an honest unavailable state', async ({page}) => {
  await page.route('**/api/v1/evidence/raw-development', (route) =>
    route.fulfill({ status: 404, json: { detail: 'Artifact unavailable.' } }),
  );
  await page.goto('/#experiment-lab');
  await expect(page.getByRole('heading', { name: 'Raw development evidence unavailable' })).toBeVisible();
  await expect(page.getByText(/no prediction or performance result is displayed/i)).toBeVisible();
});

test('packaged reviewed development artifact renders its complete real cohort', async ({page}) => {
  const health = await (await page.request.get('/health')).json();
  test.skip(health.release_class !== 'OFFLINE_RESEARCH_ENGINEERING_ONLY', 'v2 research artifact required');
  const response = await page.request.get('/api/v1/evidence/raw-development');
  expect(response.ok()).toBe(true);
  const artifact = await response.json();
  expect(artifact.rows).toHaveLength(212);
  expect(artifact.direct_checkpoint_128_sha256).toHaveLength(64);
  expect(artifact.horizons.map((horizon: {eligible_rows: number}) => horizon.eligible_rows)).toEqual([176, 140, 135]);
  await page.goto('/#experiment-lab');
  await expect(page.getByRole('link', {name:/Protocol and evidence/})).toBeVisible();
  await expect(page.getByRole('link', {name:/Frozen protocol/})).toHaveCount(0);
  const resultTable = page.getByRole('table', {name: /All TRAIN-development predictions and truths/});
  await expect(resultTable.locator('tbody tr')).toHaveCount(636);
  await expect(page.getByText('12.23123253969628', {exact: true})).toBeVisible();
  await page.screenshot({path:resolve(import.meta.dirname, '../../evidence/browser/v2-research-results.png'),fullPage:true});
});
