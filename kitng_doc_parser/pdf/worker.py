"""Cancellable, isolated PDF operations for bounded parsing."""
import asyncio
import multiprocessing
from .chunking import extract_pdf_pages, extract_pdf_raw_text, get_pdf_page_count, inspect_pdf_pages
from .raster import render_pdf_page_png, PdfRenderError
from ..budget import ParseBudget

def _child(connection, operation, data, start, end):
    try:
        if operation == "count":
            result = get_pdf_page_count(data)
        elif operation == "chunk":
            result = extract_pdf_pages(data, start, end)
        elif operation == "text":
            result = extract_pdf_raw_text(data)
        elif operation == "inspect":
            result = inspect_pdf_pages(data)
        elif operation == "render_png":
            result = render_pdf_page_png(data, start)
        else:
            raise ValueError("Unknown PDF operation")
        connection.send((True, result))
    except Exception as ex:
        connection.send((False, {"type": type(ex).__name__, "reason_code": getattr(ex, "reason_code", "renderer_failure")} if operation == "render_png" else type(ex).__name__))
    finally:
        connection.close()

async def pdf_operation(operation, data, budget, start=0, end=0):
    # PDFium is never run in the ASGI process, even for a legacy caller that did
    # not supply an overall parse budget. A renderer without a caller deadline
    # receives a bounded 90-second operation budget.
    if operation == "render_png" and budget is None:
        budget = ParseBudget.seconds(90)
    if budget is None:
        if operation == "count": return get_pdf_page_count(data)
        if operation == "chunk": return extract_pdf_pages(data, start, end)
        if operation == "inspect": return inspect_pdf_pages(data)
        return extract_pdf_raw_text(data)
    budget.check()
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(child, operation, data, start, end), daemon=True)
    receiver = None
    try:
        process.start()
        child.close()
        # A batched native pass can return megabytes. poll() only promises a
        # message header, so synchronous recv() would block the event loop while
        # its remaining bytes arrive. Terminating the child in finally closes
        # the pipe and unblocks this disposable receiver on timeout/cancel.
        receiver = asyncio.create_task(asyncio.to_thread(parent.recv))
        try:
            ok, result = await asyncio.wait_for(asyncio.shield(receiver), budget.remaining())
        except (EOFError, OSError) as ex:
            if operation == "render_png":
                raise PdfRenderError("renderer_failure") from ex
            raise ValueError("PDF worker exited without a result") from ex
        budget.check()
        if not ok:
            if operation == "render_png":
                raise PdfRenderError(result.get("reason_code", "renderer_failure"))
            raise ValueError("PDF operation failed: " + result)
        return result
    finally:
        child.close()
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            await asyncio.to_thread(process.join, 2.0)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 2.0)
        parent.close()
        if receiver is not None:
            # Consume EOF/errors from the receiver after terminating the child.
            # Do not suppress this coroutine's own cancellation/shutdown.
            await asyncio.wait({receiver}, timeout=2.0)
            if receiver.done() and not receiver.cancelled():
                receiver.exception()
