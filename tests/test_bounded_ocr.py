import asyncio
import io
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from pypdf import PdfWriter

from kitng_doc_parser import ChunkCache, DocumentParser, DocumentRef, ParseBudget, ProviderAdmission
from kitng_doc_parser.admission import ProviderCooldown
from kitng_doc_parser.ocr.openrouter import OpenRouterClient, OpenRouterError, _parse_retry_after
from kitng_doc_parser.pdf.chunking import extract_pdf_pages, extract_pdf_raw_text, get_pdf_page_count
from kitng_doc_parser.pdf.worker import pdf_operation


def pdf(note="first", pages=5):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Subject": note})
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


async def inline_pdf(operation, data, budget, start=0, end=0):
    if budget:
        budget.check()
    if operation == "count": return get_pdf_page_count(data)
    if operation == "chunk": return extract_pdf_pages(data, start, end)
    return extract_pdf_raw_text(data)


class FakeOcr:
    _api_key = "synthetic-test-key"
    model_name = "synthetic-model"
    _base_url = "https://mock.invalid/"

    def __init__(self): self.calls = []

    async def convert_to_markdown(self, data, mime, prompt=None, *, budget=None):
        self.calls.append(prompt)
        return "Recognized synthetic text."


class BoundedOcrTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_inflight_success_does_not_clear_newer_retry_after(self):
        now = [0.0]
        admission = ProviderAdmission(2, clock=lambda: now[0])
        async with admission.slot() as old_a:
            async with admission.slot() as old_b:
                admission.failed(120)
                admission.succeeded(old_b)
                self.assertEqual(admission.retry_after, 120)
        with self.assertRaises(ProviderCooldown):
            async with admission.slot():
                self.fail("Retry-After must survive the earlier in-flight success")
        now[0] = 121
        async with admission.slot() as probe:
            admission.succeeded(probe)
        self.assertEqual(admission.retry_after, 0)

    async def test_global_cooldown_retry_after_and_single_half_open_probe(self):
        now = [0.0]
        admission = ProviderAdmission(2, clock=lambda: now[0])
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={"retry-after": "30"})
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        clients = [OpenRouterClient("test", admission=admission) for _ in range(2)]
        for client in clients:
            await client._client.aclose()
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://mock.invalid/")
        try:
            with self.assertRaises(OpenRouterError) as error:
                await clients[0].convert_to_markdown(b"x", "application/pdf", budget=ParseBudget.seconds(1))
            self.assertEqual(error.exception.retry_after, 30)
            with self.assertRaises(OpenRouterError):
                await clients[1].convert_to_markdown(b"x", "application/pdf", budget=ParseBudget.seconds(1))
            self.assertEqual(len(calls), 1)
            now[0] = 31
            probe = asyncio.create_task(clients[0].convert_to_markdown(b"x", "application/pdf", budget=ParseBudget.seconds(1)))
            await entered.wait()
            with self.assertRaises(OpenRouterError):
                await clients[1].convert_to_markdown(b"x", "application/pdf", budget=ParseBudget.seconds(1))
            release.set()
            self.assertEqual(await probe, "ok")
            self.assertEqual(len(calls), 2)
        finally:
            for client in clients: await client.aclose()

    async def test_http_deadline_is_cumulative_and_not_three_attempts(self):
        calls = []

        async def handler(request):
            calls.append(request)
            await asyncio.sleep(0.3)
            return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

        client = OpenRouterClient("test", max_attempts=3)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://mock.invalid/")
        try:
            with self.assertRaises(OpenRouterError):
                await client.convert_to_markdown(b"x", "application/pdf", budget=ParseBudget.seconds(0.03))
            self.assertEqual(len(calls), 1)
        finally:
            await client.aclose()

    async def test_bounded_resume_and_content_revision_do_not_mix_chunks(self):
        client = FakeOcr()
        parser = DocumentParser(client, chunk_size=2)
        ref = DocumentRef("scope", "same.pdf", "same.pdf")
        with tempfile.TemporaryDirectory() as directory, patch("kitng_doc_parser.pipeline.pdf_operation", inline_pdf):
            cache = ChunkCache(directory)
            data = pdf()
            result = await parser.parse(ref, data, chunk_cache=cache, budget=ParseBudget.seconds(2))
            self.assertEqual((result.status, result.completed_chunks), ("partial", 1))
            self.assertEqual(len(client.calls), 1)
            for _ in range(2):
                result = await parser.parse(ref, data, chunk_cache=cache, budget=ParseBudget.seconds(2))
            self.assertEqual((result.status, len(client.calls)), ("complete", 3))
            again = await parser.parse(ref, data, chunk_cache=cache, budget=ParseBudget.seconds(2))
            self.assertEqual(len(client.calls), 3)
            changed = await parser.parse(ref, pdf("other"), chunk_cache=cache, budget=ParseBudget.seconds(2))
            self.assertEqual(changed.completed_chunks, 1)
            self.assertNotEqual(changed.processing_revision, again.processing_revision)
            self.assertEqual(len(client.calls), 4)

    async def test_pdf_worker_process_and_deadline_cleanup(self):
        self.assertEqual(await pdf_operation("count", pdf(pages=2), ParseBudget.seconds(10)), 2)
        with self.assertRaises(TimeoutError):
            await pdf_operation("count", pdf(), ParseBudget.seconds(0))

    async def test_lost_fence_after_http_does_not_save_a_chunk(self):
        client = FakeOcr()
        lost = [False]
        async def late_result(*args, **kwargs):
            lost[0] = True
            return "late fenced result"
        client.convert_to_markdown = late_result
        parser = DocumentParser(client, chunk_size=2)
        with tempfile.TemporaryDirectory() as directory, patch("kitng_doc_parser.pipeline.pdf_operation", inline_pdf):
            with self.assertRaises(asyncio.CancelledError):
                await parser.parse(DocumentRef("scope", "doc", "doc.pdf"), pdf(), chunk_cache=ChunkCache(directory),
                                   budget=ParseBudget.seconds(2, cancelled=lambda: lost[0]))
            from pathlib import Path
            self.assertEqual(list(Path(directory).rglob("chunk_*.md")), [])

    async def test_deadline_between_metadata_and_first_chunk_is_retryable_failure(self):
        client = FakeOcr()
        parser = DocumentParser(client, chunk_size=2)
        async def expires_after_metadata(operation, data, budget, start=0, end=0):
            result = await inline_pdf(operation, data, budget, start, end)
            if operation == "count":
                budget.deadline = time.monotonic() - 1
            return result
        with tempfile.TemporaryDirectory() as directory, patch("kitng_doc_parser.pipeline.pdf_operation", expires_after_metadata):
            result = await parser.parse(DocumentRef("scope", "doc", "doc.pdf"), pdf(),
                                        chunk_cache=ChunkCache(directory), budget=ParseBudget.seconds(2))
            self.assertEqual(result.error, "slice_deadline")
            self.assertTrue(result.retryable)
            self.assertEqual(result.completed_chunks, 0)
            self.assertEqual(client.calls, [])

    async def test_admission_wait_counts_against_budget(self):
        admission = ProviderAdmission(1)
        async with admission.slot():
            with self.assertRaises(TimeoutError):
                async with admission.slot(ParseBudget.seconds(0.02)):
                    self.fail("cannot acquire while occupied")
        async with admission.slot(ParseBudget.seconds(1)):
            pass

    def test_retry_after_numeric_and_http_date(self):
        self.assertEqual(_parse_retry_after("123"), 123)
        self.assertIsNotNone(_parse_retry_after("Wed, 21 Oct 2037 07:28:00 GMT"))
        self.assertIsNone(_parse_retry_after("invalid"))
