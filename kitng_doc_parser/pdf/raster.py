"""One-page PDFium rasterization, called only in the disposable PDF worker."""
from contextlib import suppress
import io
import math

MAX_RENDER_LONG_SIDE = 2000
MAX_RENDER_PIXELS = 4_000_000
MAX_RENDER_PNG_BYTES = 8 * 1024 * 1024


class PdfRenderError(ValueError):
    def __init__(self, reason_code="renderer_failure"):
        self.reason_code = reason_code
        super().__init__("PDF rasterization rejected: " + reason_code)


class _LimitedBuffer(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, value):
        if self.tell() + len(value) > self.limit:
            raise PdfRenderError("renderer_png_limit")
        return super().write(value)


def render_pdf_page_png(data: bytes, page_number: int, *,
                        max_long_side=MAX_RENDER_LONG_SIDE,
                        max_pixels=MAX_RENDER_PIXELS,
                        max_png_bytes=MAX_RENDER_PNG_BYTES) -> dict:
    """Render one original 1-based page; never return raw PDF as fallback.

    Pixel allocation is checked before PDFium allocates its bitmap and again
    afterwards. PNG encoding writes to a bounded buffer. Native handles stay
    alive until PIL has finished encoding, then close in dependency order.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as ex:
        raise PdfRenderError("renderer_unavailable") from ex
    doc = page = bitmap = image = rgb = None
    try:
        long_side = min(MAX_RENDER_LONG_SIDE, max(1, int(max_long_side)))
        pixels = min(MAX_RENDER_PIXELS, max(1, int(max_pixels)))
        png_limit = min(MAX_RENDER_PNG_BYTES, max(1, int(max_png_bytes)))
        doc = pdfium.PdfDocument(data)
        # Forms must be initialized before obtaining page handles.
        doc.init_forms()
        if not isinstance(page_number, int) or page_number < 1 or page_number > len(doc):
            raise PdfRenderError("renderer_page_out_of_range")
        page = doc[page_number - 1]
        width, height = page.get_size()
        if not all(math.isfinite(value) and value > 0 for value in (width, height)):
            raise PdfRenderError("renderer_page_dimensions")
        scale = min((long_side - 0.001) / max(width, height),
                    math.sqrt(pixels / (width * height)) * 0.999)
        if not math.isfinite(scale) or scale <= 0:
            raise PdfRenderError("renderer_page_dimensions")
        expected_w, expected_h = max(1, math.ceil(width * scale)), max(1, math.ceil(height * scale))
        if max(expected_w, expected_h) > long_side or expected_w * expected_h > pixels:
            raise PdfRenderError("renderer_pixel_limit")
        bitmap = page.render(scale=scale, rotation=0, may_draw_forms=True,
            draw_annots=True, fill_color=(255, 255, 255, 255),
            prefer_bgrx=True, rev_byteorder=True, limit_image_cache=True)
        if max(bitmap.width, bitmap.height) > long_side or bitmap.width * bitmap.height > pixels:
            raise PdfRenderError("renderer_pixel_limit")
        image = bitmap.to_pil()
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        with _LimitedBuffer(png_limit) as output:
            rgb.save(output, format="PNG")
            png = output.getvalue()
        return {"png": png, "width": bitmap.width, "height": bitmap.height,
                "page_number": page_number, "renderer": "pypdfium2"}
    except PdfRenderError:
        raise
    except Exception as ex:
        raise PdfRenderError("renderer_failure") from ex
    finally:
        if rgb is not None and rgb is not image:
            with suppress(Exception): rgb.close()
        for value in (image, bitmap, page, doc):
            if value is not None:
                with suppress(Exception): value.close()
