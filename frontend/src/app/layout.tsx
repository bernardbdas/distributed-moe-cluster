import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SYS_02 // Distributed MoE Foundation Model Cluster',
  description: 'Distributed 3D-Parallel Foundation Model Pre-Training & Alignment Cluster (32x H100)',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
