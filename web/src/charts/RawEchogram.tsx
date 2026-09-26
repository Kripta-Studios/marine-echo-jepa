import { useEffect, useRef, useState } from "react";
import type { ObservationRow } from "../api/client";

export function RawEchogram({
  rows,
  channel,
  label,
}: {
  rows: ObservationRow[];
  channel: number;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [tableOpen, setTableOpen] = useState(false);
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx || !canvas.current) return;
    ctx.fillStyle = "#132d3b";
    ctx.fillRect(0, 0, 960, 320);
    rows.forEach((row, x) =>
      row.counts[channel]?.forEach((value, y) => {
        if (value === null) ctx.fillStyle = "#536570";
        else {
          const level = Math.min(1, Math.max(0, value / 65535));
          ctx.fillStyle = `hsl(${220 - level * 210} 72% ${18 + level * 52}%)`;
        }
        ctx.fillRect(
          (x * 960) / rows.length,
          (y * 320) / 64,
          960 / rows.length + 1,
          320 / 64,
        );
      }),
    );
  }, [rows, channel]);
  return (
    <div className="raw-plot">
      <div className="plot-axis">
        <span>Sample index groups → 0–63</span>
        <strong>{label}</strong>
        <span>AZFP raw counts · uncalibrated</span>
      </div>
      <canvas
        ref={canvas}
        width={960}
        height={320}
        role="img"
        aria-label={`${label}: ${rows.length} observed time bins; raw counts, sample-index axis. Accessible table below.`}
      />
      <div className="plot-axis">
        <span>{rows[0]?.event_time_utc ?? "No observations"}</span>
        <span>{rows.at(-1)?.event_time_utc ?? ""}</span>
      </div>
      <p>
        Fixed colour scale: 0–65,535 digitizer counts. Grey cells are missing.
        Vertical groups represent instrument samples, not metres or depth.
      </p>
      <details onToggle={(event) => setTableOpen(event.currentTarget.open)}>
        <summary>Accessible observation table ({rows.length} bins)</summary>
        {tableOpen && (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>UTC bin end</th>
                  <th>Pings</th>
                  {Array.from({ length: 64 }, (_, group) => (
                    <th key={group}>Group {group} · raw counts</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.event_time_utc}>
                    <td>{row.event_time_utc}</td>
                    <td>{row.ping_count}</td>
                    {Array.from({ length: 64 }, (_, group) => (
                      <td key={group}>
                        {row.counts[channel]?.[group]?.toFixed(1) ?? "Missing"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </details>
    </div>
  );
}
