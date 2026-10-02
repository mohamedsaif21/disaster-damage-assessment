import type { LucideIcon } from "lucide-react";
import { ClipboardCheck, Cpu, Layers, TrendingUp } from "lucide-react";

import { Card, CardContent } from "@/components/ui/Card";
import {
  formatCount,
  formatDamageLevel,
  formatDateTime,
  formatPercentage,
  shortId,
} from "@/lib/format";
import type { AssessmentHistoryItem } from "@/lib/api/types";

/**
 * Summary of the authenticated user's assessment history.
 *
 * Every metric is derived from fields the history endpoint
 * actually returns. Nothing is estimated, weighted or invented.
 * The backend already orders the list newest first, so the first
 * element is the latest assessment.
 */

export interface DashboardSummaryProps {
  assessments: AssessmentHistoryItem[];
  total: number;
}

interface MetricCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
  detail: string;
}

function MetricCard({ icon: Icon, label, value, detail }: MetricCardProps) {
  return (
    <Card>
      <CardContent className="flex items-start gap-3">
        <span
          aria-hidden="true"
          className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-slate-100 text-slate-600"
        >
          <Icon className="h-4 w-4" />
        </span>

        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            {label}
          </p>
          <p className="mt-1 truncate text-2xl font-semibold tracking-tight text-slate-900">
            {value}
          </p>
          <p className="mt-1 truncate text-sm text-slate-600">{detail}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function DashboardSummary({
  assessments,
  total,
}: DashboardSummaryProps) {
  const latest = assessments[0];

  // Highest damage is a plain maximum over a real numeric field.
  // No severity weighting or threshold logic is applied here.
  const highest = assessments.reduce<AssessmentHistoryItem | undefined>(
    (best, item) =>
      best === undefined || item.damage_percentage > best.damage_percentage
        ? item
        : best,
    undefined,
  );

  const modelNames = new Set(
    assessments
      .map((item) => item.assessment_models?.name)
      .filter((name): name is string => Boolean(name)),
  );

  return (
    <section aria-label="Assessment summary">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          icon={ClipboardCheck}
          label="Total Assessments"
          value={formatCount(total)}
          detail="In your assessment history"
        />

        <MetricCard
          icon={Layers}
          label="Latest Assessment"
          value={latest ? formatDamageLevel(latest.damage_level) : "None"}
          detail={
            latest
              ? `${formatPercentage(latest.damage_percentage)} damage on ${formatDateTime(latest.created_at)}`
              : "No assessment recorded yet"
          }
        />

        <MetricCard
          icon={TrendingUp}
          label="Highest Damage Recorded"
          value={highest ? formatPercentage(highest.damage_percentage) : "None"}
          detail={
            highest
              ? `${formatDamageLevel(highest.damage_level)} on assessment ${shortId(highest.id)}`
              : "No assessment recorded yet"
          }
        />

        <MetricCard
          icon={Cpu}
          label="Models Used"
          value={formatCount(modelNames.size)}
          detail={
            modelNames.size === 0
              ? "No linked model recorded"
              : [...modelNames].join(", ")
          }
        />
      </div>
    </section>
  );
}
