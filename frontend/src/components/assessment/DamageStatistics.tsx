import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/Card";
import {
  formatCount,
  formatDamageLevel,
  formatPercentage,
} from "@/lib/format";
import type { AssessmentClassStatistic } from "@/lib/api/types";

/**
 * Pixel statistics for one assessment.
 *
 * Rows are exactly what `assessment_class_statistics` returned,
 * already sorted by `class_id`; nothing is summed, averaged or
 * re-ordered here. Class names are the backend's own stored labels
 * (from the `ClassStatistic` join), so no class name, count or
 * percentage is invented, derived or hard-coded. When the backend
 * recorded no statistics the section is omitted entirely rather
 * than rendering an empty or placeholder table.
 */

export interface DamageStatisticsProps {
  statistics: AssessmentClassStatistic[];
}

export function DamageStatistics({ statistics }: DamageStatisticsProps) {
  if (statistics.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Damage Statistics</CardTitle>
        <p className="mt-1 text-sm text-slate-500">
          Pixel counts per class, as returned by the assessment.
        </p>
      </CardHeader>

      <CardContent>
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th
                scope="col"
                className="py-2 pr-4 text-xs font-medium uppercase tracking-wide text-slate-500"
              >
                Class
              </th>
              <th
                scope="col"
                className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wide text-slate-500"
              >
                Pixels
              </th>
              <th
                scope="col"
                className="py-2 text-right text-xs font-medium uppercase tracking-wide text-slate-500"
              >
                Share
              </th>
            </tr>
          </thead>

          <tbody>
            {statistics.map((statistic) => (
              <tr
                key={statistic.id}
                className="border-b border-slate-100 last:border-b-0"
              >
                <th
                  scope="row"
                  className="py-2 pr-4 text-left font-medium text-slate-900"
                >
                  {formatDamageLevel(statistic.class_name)}
                </th>
                <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-slate-700">
                  {formatCount(statistic.pixel_count)}
                </td>
                <td className="whitespace-nowrap py-2 text-right tabular-nums text-slate-700">
                  {formatPercentage(statistic.percentage)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
