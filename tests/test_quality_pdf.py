import asyncio
import io
import gzip
import hashlib
import multiprocessing
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, NumberObject, DecodedStreamObject

from kitng_doc_parser import (ChunkCache, DocumentParser, DocumentRef, ParseBudget,
                              validate_markdown_quality, QUALITY_POLICY)
from kitng_doc_parser.pdf.chunking import inspect_pdf_pages, extract_pdf_pages
from kitng_doc_parser.pdf.worker import pdf_operation
from kitng_doc_parser.ocr.openrouter import OpenRouterClient, OpenRouterError

TEXT = "Engineering design section 1.2. Pressure is 123.45 kPa; dimensions 100 x 200 mm. Preserve all references and measurements."


def synthetic_pdf(kinds):
    writer = PdfWriter()
    for kind in kinds:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        resources = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        command = b''
        if kind in {'text', 'image_with_text'}:
            command += b'BT /F1 10 Tf 20 700 Td (' + TEXT.encode() + b') Tj ET\n'
        if kind in {'scan', 'image_with_text'}:
            image = DecodedStreamObject()
            image.set_data(b'\xff\xff\xff')
            image.update({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Image'), NameObject('/Width'): NumberObject(1), NameObject('/Height'): NumberObject(1), NameObject('/ColorSpace'): NameObject('/DeviceRGB'), NameObject('/BitsPerComponent'): NumberObject(8)})
            resources[NameObject('/XObject')] = DictionaryObject({NameObject('/Im0'): writer._add_object(image)})
            command += b'q 600 0 0 800 0 0 cm /Im0 Do Q\n'
        page[NameObject('/Resources')] = resources
        stream = DecodedStreamObject()
        stream.set_data(command)
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class FakeOcr:
    _api_key = 'test'
    _base_url = 'https://synthetic.invalid/'
    model_name = 'synthetic'
    def __init__(self, output=TEXT): self.calls, self.output = [], output
    async def convert_to_markdown(self, data, mime, prompt=None, **kwargs):
        from PIL import Image
        assert mime == 'image/png'
        with Image.open(io.BytesIO(data)) as image:
            assert image.format == 'PNG'
            assert max(image.size) <= 2000 and image.width * image.height <= 4_000_000
        self.calls.append(1)
        return self.output


class QualityTests(unittest.TestCase):
    def test_exact_million_dot_regression_and_empty_table(self):
        checked = validate_markdown_quality('Engineering section 1.2\n' + '.' * 1_046_400)
        self.assertFalse(checked.valid)
        self.assertEqual(checked.reason_code, 'repeated_character_run')
        self.assertFalse(validate_markdown_quality('Header\n' + '| | | |\n' * 200).valid)

    def test_real_leaders_numbers_and_filled_table_preserved(self):
        text = 'Section 1.2 ' + '.' * 162 + ' 175\n| Item | Amount |\n|---|---|\n| 01.002 | 123.45 |\n'
        before = text
        self.assertTrue(validate_markdown_quality(text, page_count=1).valid)
        self.assertEqual(text, before)

    def test_metadata_and_missing_page_comments_do_not_make_content(self):
        self.assertFalse(validate_markdown_quality('---\ntitle: "fake"\n---\n<!-- Pages 1 - 1: не распознано -->').valid)
        self.assertTrue(validate_markdown_quality(TEXT + '\n<!-- Pages 2 - 2: не распознано -->').valid)

    def test_truncation_and_size_are_not_silently_accepted(self):
        self.assertEqual(validate_markdown_quality(TEXT, finish_reason='length').reason_code, 'truncated_output')
        self.assertFalse(validate_markdown_quality(TEXT, require_finish_reason=True).valid)
        self.assertFalse(validate_markdown_quality(TEXT, max_chars=10).valid)


class PipelineQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_native_in_one_inspection_and_zero_paid_calls(self):
        client = FakeOcr()
        parser = DocumentParser(client, chunk_size=6)
        calls = []
        async def inspect_only(operation, data, budget, start=0, end=0):
            calls.append(operation)
            if budget: budget.check()
            self.assertEqual(operation, 'inspect')
            return inspect_pdf_pages(data)
        with tempfile.TemporaryDirectory() as directory, patch('kitng_doc_parser.pipeline.pdf_operation', inspect_only):
            result = await parser.parse(DocumentRef('test', 'native.pdf', 'native.pdf'), synthetic_pdf(['text'] * 175), chunk_cache=ChunkCache(directory), budget=ParseBudget.seconds(10))
        self.assertEqual((result.status, result.origin, result.native_pages, result.ocr_pages), ('complete', 'native_pdf', 175, 0))
        self.assertEqual((result.completed_chunks, result.total_chunks), (175, 175))
        self.assertEqual(calls, ['inspect'])
        self.assertEqual(client.calls, [])

    async def test_mixed_document_native_checkpoints_and_one_page_ocr_resume(self):
        client = FakeOcr()
        parser = DocumentParser(client, chunk_size=6)
        data = synthetic_pdf(['text', 'scan', 'text', 'scan'])
        ref = DocumentRef('test', 'mixed.pdf', 'mixed.pdf')
        with tempfile.TemporaryDirectory() as directory:
            cache = ChunkCache(directory)
            first = await parser.parse(ref, data, chunk_cache=cache, budget=ParseBudget.seconds(20))
            self.assertEqual((first.status, first.completed_chunks, first.native_pages, first.ocr_pages), ('partial', 3, 2, 1))
            second = await parser.parse(ref, data, chunk_cache=cache, budget=ParseBudget.seconds(20))
            self.assertEqual((second.status, second.origin, second.completed_chunks), ('complete', 'mixed_pdf', 4))
            self.assertEqual(second.page_origins, {1:'native_pdf', 2:'ocr', 3:'native_pdf', 4:'ocr'})
            self.assertEqual(client.calls, [1, 1])
            await parser.parse(ref, data, chunk_cache=cache, budget=ParseBudget.seconds(10))
            self.assertEqual(client.calls, [1, 1])

    async def test_no_key_native_complete_but_scan_is_not_complete(self):
        parser = DocumentParser()
        native = await parser.parse(DocumentRef('s', 'n.pdf', 'n.pdf'), synthetic_pdf(['text']))
        self.assertEqual((native.status, native.origin), ('complete', 'native_pdf'))
        mixed = await parser.parse(DocumentRef('s', 'm.pdf', 'm.pdf'), synthetic_pdf(['text', 'scan']))
        self.assertEqual((mixed.status, mixed.reason_code, mixed.native_pages), ('partial', 'ocr_not_configured', 1))

    async def test_raster_page_even_with_sparse_overlay_uses_ocr(self):
        client = FakeOcr()
        result = await DocumentParser(client).parse(DocumentRef('s','s.pdf','s.pdf'), synthetic_pdf(['image_with_text']))
        self.assertEqual((result.origin, client.calls), ('ocr', [1]))

    async def test_corrupt_checkpoint_is_rejected_on_reuse_before_assembly(self):
        client = FakeOcr()
        parser = DocumentParser(client)
        ref, data = DocumentRef('s', 'scan.pdf', 'scan.pdf'), synthetic_pdf(['scan'])
        with tempfile.TemporaryDirectory() as directory:
            cache = ChunkCache(directory)
            first = await parser.parse(ref, data, chunk_cache=cache)
            rev = cache.for_revision(first.processing_revision)
            # Simulate pre-fix checkpoint bytes; current save_chunk rejects them.
            (rev._doc_dir(ref.scope, ref.item_id) / 'chunk_0.md').write_text('Header' + '.' * 1_046_400)
            again = await parser.parse(ref, data, chunk_cache=cache)
            self.assertEqual((again.status, len(client.calls)), ('complete', 2))
            self.assertNotIn('.' * 512, again.markdown)

    async def test_invalid_ocr_is_never_written_as_done(self):
        client = FakeOcr('Header' + '.' * 1_046_400)
        parser = DocumentParser(client)
        ref, data = DocumentRef('s','s.pdf','s.pdf'), synthetic_pdf(['scan'])
        with tempfile.TemporaryDirectory() as directory:
            result = await parser.parse(ref, data, chunk_cache=ChunkCache(directory))
            self.assertEqual((result.status, result.reason_code, result.retryable, result.completed_chunks), ('partial','invalid_content',False,0))
            self.assertEqual(list(Path(directory).rglob('chunk_*.md')), [])

    async def test_previous_quality_policy_manifest_is_never_reused(self):
        client = FakeOcr()
        parser = DocumentParser(client)
        ref, data = DocumentRef('s','old.pdf','old.pdf'), synthetic_pdf(['scan'])
        with tempfile.TemporaryDirectory() as directory:
            cache = ChunkCache(directory)
            rev = cache.for_revision(parser.processing_revision(data))
            rev.save_chunk('s','old.pdf',0,'Previous policy OCR text must not be reused.')
            rev.save_manifest('s','old.pdf',{'quality_policy':'previous-policy', 'source_revision':hashlib.sha256(data).hexdigest(), 'analysis_complete':True, 'total_pages':1, 'chunks':[{'index':0,'status':'done','origin':'ocr'}]})
            result = await parser.parse(ref,data,chunk_cache=cache)
            self.assertEqual(client.calls,[1])
            self.assertNotIn('Previous policy OCR text',result.markdown)
            self.assertEqual(result.quality_policy,QUALITY_POLICY)

    async def test_legacy_direct_checkpoint_read_rejects_million_dots_without_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ChunkCache(directory)
            location = cache._doc_dir('s','legacy.pdf')
            location.mkdir(parents=True)
            original = 'Legacy OCR header\n' + '.' * 1_046_400
            path = location / 'chunk_0.md'
            path.write_text(original)
            self.assertIsNone(cache.get_chunk('s','legacy.pdf',0))
            self.assertEqual(path.read_text(),original)

    async def test_native_inspection_is_cancellable_and_real_worker_bounded(self):
        data = synthetic_pdf(['text'] * 3)
        info = await pdf_operation('inspect', data, ParseBudget.seconds(10))
        self.assertEqual(info['total_pages'], 3)
        with self.assertRaises(TimeoutError):
            await pdf_operation('inspect', data, ParseBudget.seconds(0))

    async def test_cancelling_active_native_worker_leaves_no_child(self):
        before = {process.pid for process in multiprocessing.active_children()}
        task = asyncio.create_task(pdf_operation('inspect',synthetic_pdf(['text'] * 500),ParseBudget.seconds(60)))
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task,5)
        remaining = {process.pid for process in multiprocessing.active_children()} - before
        self.assertEqual(remaining,set())


class HttpQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_gzip_body_is_decoded_once(self):
        payload = json.dumps({'choices':[{'finish_reason':'stop','message':{'content':TEXT}}]}).encode()
        compressed = gzip.compress(payload)
        response = httpx.Response(200, headers={'content-encoding':'gzip','content-length':str(len(compressed)), 'content-type':'application/json'}, stream=httpx.ByteStream(compressed))
        result, _ = await self._request(response)
        self.assertEqual(result, TEXT)

    async def test_429_default_cooldown_and_explicit_retry_after(self):
        from kitng_doc_parser import ProviderAdmission
        for header, expected in [(None, 60), ('7', 7), ('0', 0)]:
            admission = ProviderAdmission(clock=lambda: 0)
            client = OpenRouterClient('synthetic', admission=admission)
            await client._client.aclose()
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429, headers={} if header is None else {'retry-after':header})), base_url='https://mock.invalid/')
            try:
                with self.assertRaises(OpenRouterError) as error:
                    await client.convert_to_markdown(b'x','application/pdf',budget=ParseBudget.seconds(1))
                self.assertEqual(error.exception.retry_after, expected)
                self.assertEqual(error.exception.reason_code, 'provider_rate_limited')
            finally: await client.aclose()

    async def test_retry_after_does_not_sleep_beyond_cumulative_budget(self):
        client = OpenRouterClient('synthetic')
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429, headers={'retry-after':'120'})), base_url='https://mock.invalid/')
        try:
            with self.assertRaises(OpenRouterError) as error:
                await asyncio.wait_for(client.convert_to_markdown(b'x','application/pdf',budget=ParseBudget.seconds(0.2, max_attempts=3)), 1)
            self.assertEqual(error.exception.reason_code, 'provider_rate_limited')
            self.assertGreater(error.exception.retry_after, 100)
        finally: await client.aclose()

    async def _request(self, response, **options):
        client = OpenRouterClient('synthetic', **options)
        await client._client.aclose()
        payloads = []
        async def handler(request):
            payloads.append(json.loads(request.content))
            return response
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='https://mock.invalid/')
        try:
            result = await client.convert_to_markdown(b'synthetic', 'application/pdf', budget=ParseBudget.seconds(2))
            return result, payloads
        finally: await client.aclose()

    async def test_finish_length_rejected_with_no_retry(self):
        response = httpx.Response(200, json={'choices':[{'finish_reason':'length','message':{'content':TEXT}}]})
        with self.assertRaises(OpenRouterError) as error:
            await self._request(response)
        self.assertEqual((error.exception.reason_code, error.exception.retryable), ('invalid_content', False))

    async def test_max_tokens_sent_and_valid_numbers_untouched(self):
        response = httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{'content':TEXT}}], 'usage':{'completion_tokens':40}})
        result, payloads = await self._request(response)
        self.assertEqual(result, TEXT)
        self.assertEqual(payloads[0]['max_tokens'], 16384)

    async def test_oversize_response_and_empty_table_rejected(self):
        for response in [httpx.Response(200, content=b'x' * 2000), httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{'content':'Heading\n' + '| | | |\n' * 200}}]})]:
            with self.assertRaises(OpenRouterError) as error:
                await self._request(response, max_response_bytes=1000)
            self.assertEqual(error.exception.reason_code, 'invalid_content')

    async def test_timeout_classification(self):
        client = OpenRouterClient('synthetic', retry_base_seconds=0)
        await client._client.aclose()
        async def handler(request): raise httpx.ReadTimeout('synthetic')
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='https://mock.invalid/')
        try:
            with self.assertRaises(OpenRouterError) as error:
                await client.convert_to_markdown(b'x','application/pdf',budget=ParseBudget.seconds(1))
            self.assertEqual(error.exception.reason_code, 'provider_timeout')
        finally: await client.aclose()
