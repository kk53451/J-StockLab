"use client";

import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";
import { TrendingUp } from "lucide-react";

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-950/80 backdrop-blur-sm">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl">
          <TrendingUp className="w-6 h-6 text-blue-600" />
          <span>J-StockLab</span>
        </Link>
        <nav className="flex items-center gap-6">
          <Link
            href="/"
            className="text-sm font-medium hover:text-blue-600 transition-colors"
          >
            대시보드
          </Link>
          <Link
            href="/stocks"
            className="text-sm font-medium hover:text-blue-600 transition-colors"
          >
            종목 목록
          </Link>
          <Link
            href="/indicators"
            className="text-sm font-medium hover:text-blue-600 transition-colors"
          >
            경제 지표
          </Link>
          <Link
            href="/compare"
            className="text-sm font-medium hover:text-blue-600 transition-colors"
          >
            종목 비교
          </Link>
          <Link
            href="/models"
            className="text-sm font-medium hover:text-blue-600 transition-colors"
          >
            모델 비교
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
