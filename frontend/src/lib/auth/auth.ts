/**
 * Typed authentication helpers over the Supabase browser client.
 *
 * This module covers authentication only. It creates no UI and
 * adds no caching layer. Every function mirrors Supabase's own
 * `{ data, error }` result: an authentication error is returned
 * to the caller rather than swallowed.
 */

import type { AuthError, Session, User } from "@supabase/supabase-js";

import { getSupabaseClient } from "../supabase/client";

export interface AuthResult<T> {
  data: T | null;
  error: AuthError | null;
}

export interface AuthUserPayload {
  user: User | null;
  session: Session | null;
}

/**
 * Create a new Supabase Auth user.
 *
 * When email confirmation is enabled a user is returned without
 * a session until the address is confirmed.
 */
export async function signUp(
  email: string,
  password: string,
): Promise<AuthResult<AuthUserPayload>> {
  const { data, error } =
    await getSupabaseClient().auth.signUp({
      email,
      password,
    });

  return { data: error ? null : data, error };
}

/**
 * Sign in with email and password.
 */
export async function signIn(
  email: string,
  password: string,
): Promise<AuthResult<AuthUserPayload>> {
  const { data, error } =
    await getSupabaseClient().auth.signInWithPassword({
      email,
      password,
    });

  return { data: error ? null : data, error };
}

/**
 * Sign out the current session and clear it from storage.
 */
export async function signOut(): Promise<AuthResult<null>> {
  const { error } = await getSupabaseClient().auth.signOut();

  return { data: null, error };
}

/**
 * Read the current session. Supabase refreshes an expired
 * token internally, so this never triggers a refresh loop.
 */
export async function getSession(): Promise<AuthResult<Session>> {
  const { data, error } =
    await getSupabaseClient().auth.getSession();

  return { data: error ? null : data.session, error };
}

/**
 * Read the current authenticated user.
 */
export async function getUser(): Promise<AuthResult<User>> {
  const { data, error } = await getSupabaseClient().auth.getUser();

  return { data: error ? null : data.user, error };
}
