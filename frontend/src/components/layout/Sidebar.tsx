"use client";

import type { LucideIcon } from "lucide-react";
import { ClipboardList, FilePlus, LayoutDashboard } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

interface NavigationItem {
  name: string;
  href: string;
  icon: LucideIcon;
}

const navigation: NavigationItem[] = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "New Assessment", href: "/assessment/new", icon: FilePlus },
  { name: "Assessment History", href: "/history", icon: ClipboardList },
];

const linkBase =
  "flex items-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2";

function isCurrentPath(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="flex h-16 shrink-0 items-center border-b border-slate-200 px-6">
        <Link
          href="/dashboard"
          className="text-sm font-semibold tracking-tight text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        >
          Disaster Damage Assessment
        </Link>
      </div>

      <nav aria-label="Primary" className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = isCurrentPath(pathname, item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                linkBase,
                "border-l-2 px-3 py-2",
                isActive
                  ? "border-slate-900 bg-slate-100 text-slate-900"
                  : "border-transparent text-slate-600 hover:bg-slate-50 hover:text-slate-900",
              )}
            >
              <Icon aria-hidden="true" className="mr-3 h-4 w-4 shrink-0" />
              {item.name}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

export function MobileNavigation() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Primary mobile"
      className="flex shrink-0 gap-2 overflow-x-auto border-b border-slate-200 bg-white px-3 py-2 md:hidden"
    >
      {navigation.map((item) => {
        const Icon = item.icon;
        const isActive = isCurrentPath(pathname, item.href);

        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              linkBase,
              "shrink-0 gap-2 border px-3 py-1.5",
              isActive
                ? "border-slate-300 bg-slate-100 text-slate-900"
                : "border-transparent text-slate-600",
            )}
          >
            <Icon aria-hidden="true" className="h-4 w-4 shrink-0" />
            {item.name}
          </Link>
        );
      })}
    </nav>
  );
}
