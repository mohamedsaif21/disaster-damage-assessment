import { DashboardView } from "@/components/dashboard/DashboardView";

/**
 * Dashboard route.
 *
 * The application shell (sidebar and header) is provided by the
 * root layout, so this page only renders the dashboard content.
 */
export default function DashboardPage() {
  return <DashboardView />;
}
