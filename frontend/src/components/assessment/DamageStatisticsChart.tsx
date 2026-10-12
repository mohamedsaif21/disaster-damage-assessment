"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3 } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  formatCount,
  formatDamageLevel,
  formatPercentage,
} from "@/lib/format";
import type { AssessmentClassStatistic } from "@/lib/api/types";

/**
 * Visual summary of the class statistics already returned by the
 * assessment detail API.
 *
 * Every value comes straight from `assessment_class_statistics`:
 * `pixel_count` is the backend's stored pixel tally and
 * `percentage` is the backend's own computed share. Nothing is
 * summed, re-derived or re-ordered here, and the rows keep the
 * backend order (ascending `class_id`), matching the table below.
 *
 * A single horizontal bar chart is used on purpose: the data is
 * one categorical field (class) against a single numeric measure
 * (pixels), so a second chart would only restate the same five
 * values. Horizontal bars keep the multi-word class labels readable
 * without rotated text on narrow screens. Meaning never relies on
 * colour alone: every bar is labelled with its value, both axes are
 * rendered, the tooltip states the value and share in words, and the
 * existing `DamageStatistics` table offers the exact figures.
 *
 * The chart only mounts in the browser once the parent has real
 * data, so no chart markup is server-rendered and no assessment
 * request is made here.
 */

const BAR_COLOR = "#334155";
const GRID_COLOR = "#e2e8f0";
const AXIS_COLOR = "#64748b";
const ROW_HEIGHT = 48;
const MIN_CHART_HEIGHT = 220;

interface ChartDatum {
  key: string;
  label: string;
  pixel_count: number;
  percentage: number;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: ChartDatum }>;
}

function toChartDatum(statistic: AssessmentClassStatistic): ChartDatum {
  return {
    key: statistic.id,
    label: formatDamageLevel(statistic.class_name),
    pixel_count: Number.isFinite(statistic.pixel_count)
      ? statistic.pixel_count
      : 0,
    percentage: Number.isFinite(statistic.percentage)
      ? statistic.percentage
      : 0,
  };
}

function ChartTooltip({ active, payload }: ChartTooltipProps) {
  const datum = payload?.[0]?.payload;

  if (!active || !datum) {
    return null;
  }

  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs shadow-sm">
      <p className="font-medium text-slate-900">{datum.label}</p>
      <p className="mt-1 tabular-nums text-slate-700">
        {formatCount(datum.pixel_count)} pixels
      </p>
      <p className="tabular-nums text-slate-500">
        {formatPercentage(datum.percentage)} of all pixels
      </p>
    </div>
  );
}

export interface DamageStatisticsChartProps {
  statistics: AssessmentClassStatistic[];
}

export function DamageStatisticsChart({
  statistics,
}: DamageStatisticsChartProps) {
  const data = statistics.map(toChartDatum);

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Class Distribution</CardTitle>
          <p className="mt-1 text-sm text-slate-500">
            Pixel share per damage class.
          </p>
        </CardHeader>

        <CardContent>
          <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-slate-300 bg-slate-50 px-6 py-12 text-center">
            <BarChart3 aria-hidden="true" className="h-5 w-5 text-slate-400" />
            <p className="text-sm font-medium text-slate-500">
              Class distribution is unavailable for this assessment.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const domainMax = Math.max(1, ...data.map((datum) => datum.pixel_count));
  const chartHeight = Math.max(MIN_CHART_HEIGHT, data.length * ROW_HEIGHT);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Class Distribution</CardTitle>
        <p className="mt-1 text-sm text-slate-500">
          Pixels per damage class, from the assessment result. Exact values are
          listed in the table below.
        </p>
      </CardHeader>

      <CardContent>
        <div className="w-full min-w-0" style={{ height: chartHeight }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{ top: 4, right: 56, bottom: 8, left: 8 }}
              desc="Bar chart of the number of pixels recorded for each damage class."
            >
              <CartesianGrid
                horizontal={false}
                stroke={GRID_COLOR}
                strokeDasharray="3 3"
              />
              <XAxis
                type="number"
                domain={[0, domainMax]}
                allowDecimals={false}
                stroke={GRID_COLOR}
                tick={{ fill: AXIS_COLOR, fontSize: 12 }}
                tickFormatter={(value) => formatCount(Number(value))}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={96}
                stroke={GRID_COLOR}
                tick={{ fill: AXIS_COLOR, fontSize: 12 }}
              />
              <Tooltip
                content={<ChartTooltip />}
                cursor={{ fill: "rgba(148, 163, 184, 0.15)" }}
              />
              <Bar
                dataKey="pixel_count"
                name="Pixels"
                fill={BAR_COLOR}
                radius={[0, 4, 4, 0]}
                isAnimationActive={false}
              >
                <LabelList
                  dataKey="pixel_count"
                  position="right"
                  fill={AXIS_COLOR}
                  fontSize={12}
                  formatter={(value) => formatCount(Number(value))}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
