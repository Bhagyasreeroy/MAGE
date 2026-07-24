import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * middleware.ts
 * ─────────────
 * Protects /dashboard routes. Reads the `mage_token` cookie that we set
 * after login. If absent, redirects to /signin.
 */
export function middleware(request: NextRequest) {
  const token = request.cookies.get('mage_token')?.value;
  const { pathname } = request.nextUrl;

  // If trying to access dashboard without a token → redirect to signin
  if (pathname.startsWith('/dashboard') && !token) {
    const signinUrl = new URL('/signin', request.url);
    signinUrl.searchParams.set('from', pathname);
    return NextResponse.redirect(signinUrl);
  }

  // If already logged in and visiting auth pages → redirect to dashboard
  if ((pathname === '/signin' || pathname === '/signup') && token) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/signin', '/signup'],
};
