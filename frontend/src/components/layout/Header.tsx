"use client";

import type { User } from "@supabase/supabase-js";
import { useEffect, useState } from "react";

import { getUser } from "@/lib/auth/auth";

export function Header() {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    let active = true;

    getUser()
      .then(({ data }) => {
        if (active) {
          setUser(data);
        }
      })
      .catch(() => {
        if (active) {
          setUser(null);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const email = user?.email ?? null;

  return (
    <header className="flex h-16 shrink-0 items-center justify-between gap-4 border-b border-slate-200 bg-white px-4 sm:px-6">
      <div className="min-w-0">
        <p className="truncate text-base font-semibold text-slate-900 sm:text-lg">
          AI Disaster Damage Assessment
        </p>
        <p className="truncate text-xs text-slate-500">
          Damage assessment and decision support
        </p>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <span
          aria-hidden="true"
          className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-200 text-xs font-semibold text-slate-700"
        >
          {email ? email.charAt(0).toUpperCase() : "-"}
        </span>
        <span className="text-sm text-slate-600">{email ?? "Signed out"}</span>
      </div>
    </header>
  );
}
