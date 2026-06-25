"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, createContext, useContext, type ReactNode } from "react";

interface UIContextType {
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

const UIContext = createContext<UIContextType>({
  sidebarCollapsed: false,
  toggleSidebar: () => {},
});

export function useUI() {
  return useContext(UIContext);
}

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <QueryClientProvider client={queryClient}>
      <UIContext.Provider
        value={{
          sidebarCollapsed,
          toggleSidebar: () => setSidebarCollapsed((v) => !v),
        }}
      >
        {children}
      </UIContext.Provider>
    </QueryClientProvider>
  );
}
