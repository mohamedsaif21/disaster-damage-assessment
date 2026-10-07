import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { formatCount, orDash } from "@/lib/format";
import type { RegisteredModelInfo } from "@/lib/api/types";

/**
 * The registered analysis model behind one assessment.
 *
 * Rendered only when `assessment_models` is present, and only with
 * the fields the backend returned: no model is inferred from
 * `model_id` alone and nothing is fetched separately. Nullable
 * values (`epoch`, `validation_loss`) fall back to the shared em
 * dash rather than showing "null" or a fabricated number.
 */

export interface ModelInformationProps {
  model: RegisteredModelInfo | null;
}

function formatValidationLoss(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return orDash(null);
  }

  return value.toFixed(4);
}

export function ModelInformation({ model }: ModelInformationProps) {
  if (!model) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Information</CardTitle>
        <p className="mt-1 text-sm text-slate-500">
          The model registered for this assessment.
        </p>
      </CardHeader>

      <CardContent>
        <dl className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2">
          <div>
            <dt className="text-xs font-medium text-slate-500">Model name</dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {orDash(model.name)}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">
              Architecture
            </dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {orDash(model.architecture)}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">Checkpoint</dt>
            <dd className="mt-0.5 break-all font-mono text-sm text-slate-900">
              {orDash(model.checkpoint)}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">Epoch</dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {model.epoch !== null ? formatCount(model.epoch) : orDash(null)}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">
              Input channels
            </dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {formatCount(model.input_channels)}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">
              Output classes
            </dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {formatCount(model.output_classes)}
            </dd>
          </div>

          <div className="sm:col-span-2">
            <dt className="text-xs font-medium text-slate-500">
              Validation loss
            </dt>
            <dd className="mt-0.5 text-sm font-medium tabular-nums text-slate-900">
              {formatValidationLoss(model.validation_loss)}
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
