"use client"
import { AppRouterCacheProvider } from '@mui/material-nextjs/v14-appRouter';
import { createTheme } from '@mui/material/styles';
import "./globals.css";
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../core/theme'


export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <AppRouterCacheProvider>
          <ThemeProvider theme={createTheme(theme)}>
            {children}
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}
