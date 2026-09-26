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
  await expect(page.getByRole('cell', {name:'NOT_RUN'}).first()).toBeVisible();
  await expect(page.getByText('Required run registry (25 runs)')).toBeVisible();
  await page.getByRole('link', {name:/Evidence & transfer/}).click();
  await expect(page.getByText('ENGINEERING_DEMO_ONLY', {exact:true})).toBeVisible();
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
