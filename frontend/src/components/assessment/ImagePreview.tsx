import Image from "next/image";

import { cn } from "@/lib/utils";

/**
 * Renders the local preview of a selected image.
 *
 * The source is a browser object URL created by `useObjectUrl`, so
 * the preview never triggers a request to FastAPI or to Supabase
 * Storage. `next/image` renders blob sources unoptimized, which is
 * the correct behaviour for a local file: there is no remote
 * optimizer to call.
 */

export interface ImagePreviewProps {
  previewUrl: string;
  alt: string;
  className?: string;
}

export function ImagePreview({
  previewUrl,
  alt,
  className,
}: ImagePreviewProps) {
  return (
    <div
      className={cn(
        "relative aspect-[4/3] w-full overflow-hidden bg-slate-100",
        className,
      )}
    >
      <Image
        src={previewUrl}
        alt={alt}
        fill
        unoptimized
        sizes="(max-width: 1023px) 100vw, 40vw"
        className="object-cover"
      />
    </div>
  );
}