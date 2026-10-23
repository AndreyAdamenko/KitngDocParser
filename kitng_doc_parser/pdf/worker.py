"""Cancellable, isolated PDF operations for bounded parsing."""
import asyncio
import multiprocessing
from .chunking import extract_pdf_pages, extract_pdf_raw_text, get_pdf_page_count

def _child(connection, operation, data, start, end):
    try:
        if operation == "count":
            result = get_pdf_page_count(data)
        elif operation == "chunk":
            result = extract_pdf_pages(data, start, end)
        elif operation == "text":
            result = extract_pdf_raw_text(data)
        else:
            raise ValueError("Unknown PDF operation")
        connection.send((True, result))
    except Exception as ex:
        connection.send((False, type(ex).__name__))
    finally:
        connection.close()

async def pdf_operation(operation, data, budget, start=0, end=0):
    if budget is None:
        if operation == "count": return get_pdf_page_count(data)
        if operation == "chunk": return extract_pdf_pages(data, start, end)
        return extract_pdf_raw_text(data)
    budget.check()
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(child, operation, data, start, end), daemon=True)
    try:
        process.start()
        child.close()
        # poll, rather than an indefinitely blocked recv thread, bounds cleanup.
        while not parent.poll():
            budget.check()
            if not process.is_alive():
                raise ValueError("PDF worker exited without a result")
            await asyncio.sleep(min(0.02, budget.remaining()))
        ok, result = parent.recv()
        budget.check()
        if not ok:
            raise ValueError("PDF operation failed: " + result)
        return result
    finally:
        parent.close()
        child.close()
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            await asyncio.to_thread(process.join, 2.0)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 2.0)
