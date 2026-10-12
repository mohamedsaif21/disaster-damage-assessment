import { HistoryView } from "@/components/history/HistoryView";

/**
 * Assessment history route.
 *
 * The application shell (sidebar and header) is provided by the
 * root layout, so this page only renders the history content. A thin
 * server component keeps the Supabase session and the history
 * request inside the client view.
 */
export default function HistoryPage() {
  return <HistoryView />;
}
