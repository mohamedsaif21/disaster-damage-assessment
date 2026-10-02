import { Card } from "@/components/ui/Card";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
          Dashboard
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Dashboard metrics, trends, and assessment summaries will be
          implemented in the next step.
        </p>
      </div>

      <Card className="p-6">
        <p className="text-sm text-slate-600">
          Application shell is in place. No assessment data is requested at this
          stage.
        </p>
      </Card>
    </div>
  );
}
