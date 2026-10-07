import { AssessmentResultView } from "@/components/assessment/AssessmentResultView";

/**
 * Assessment result page.
 *
 * A thin server component: the id arrives from the route segment
 * and is forwarded to the client container as an opaque string
 * (the backend treats it as such). All data access, loading,
 * error and retry behaviour lives in `AssessmentResultView`.
 */
interface AssessmentDetailPageProps {
  params: Promise<{
    id: string;
  }>;
}

export default async function AssessmentDetailPage({
  params,
}: AssessmentDetailPageProps) {
  const { id } = await params;

  return <AssessmentResultView key={id} assessmentId={id} />;
}