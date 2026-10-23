import asyncio
import io
import multiprocessing
import unittest
from unittest.mock import patch

from PIL import Image, ImageStat
from pypdf import PdfWriter

from kitng_doc_parser import DocumentParser, DocumentRef, ParseBudget, QUALITY_POLICY
from kitng_doc_parser.pdf.raster import render_pdf_page_png, PdfRenderError
from kitng_doc_parser.pdf.worker import pdf_operation
from test_quality_pdf import synthetic_pdf, TEXT


class RasterTests(unittest.TestCase):
    def test_real_pdfium_page_selection_dimensions_png_and_text_pixels(self):
        data = synthetic_pdf(['scan','text'])
        selected = render_pdf_page_png(data,2)
        self.assertEqual(selected['page_number'],2)
        self.assertLessEqual(len(selected['png']),8 * 1024 * 1024)
        with Image.open(io.BytesIO(selected['png'])) as image:
            self.assertEqual(image.format,'PNG')
            self.assertEqual(image.mode,'RGB')
            self.assertLessEqual(max(image.size),2000)
            self.assertLessEqual(image.width * image.height,4_000_000)
            self.assertEqual((image.width,image.height),(selected['width'],selected['height']))
            self.assertLess(min(ImageStat.Stat(image).mean),255)

    def test_huge_and_rotated_pages_are_bounded_before_render(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=100_000,height=20_000)
        page.rotate(90)
        output = io.BytesIO()
        writer.write(output)
        selected = render_pdf_page_png(output.getvalue(),1)
        self.assertLessEqual(max(selected['width'],selected['height']),2000)
        self.assertLessEqual(selected['width'] * selected['height'],4_000_000)
        self.assertGreater(selected['height'],selected['width'])

    def test_invalid_pdf_page_and_png_byte_ceiling_fail_closed(self):
        for data, page in [(b'not a PDF',1),(synthetic_pdf(['text']),2)]:
            with self.assertRaises(PdfRenderError): render_pdf_page_png(data,page)
        with self.assertRaises(PdfRenderError) as error:
            render_pdf_page_png(synthetic_pdf(['text']),1,max_png_bytes=16)
        self.assertEqual(error.exception.reason_code,'renderer_png_limit')

    def test_processing_revision_binds_raster_input_policy(self):
        parser = DocumentParser()
        self.assertIn('raster',QUALITY_POLICY)
        from kitng_doc_parser import pipeline
        before = parser.processing_revision(b'synthetic')
        with patch.object(pipeline,'MAX_RENDER_LONG_SIDE',1000):
            self.assertNotEqual(parser.processing_revision(b'synthetic'),before)


class RasterWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_renderer_runs_in_disposable_worker_with_deadline(self):
        selected = await pdf_operation('render_png',synthetic_pdf(['text']),ParseBudget.seconds(10),1,1)
        self.assertEqual(selected['page_number'],1)
        self.assertTrue(selected['png'].startswith(b'\x89PNG\r\n\x1a\n'))
        with self.assertRaises(TimeoutError):
            await pdf_operation('render_png',synthetic_pdf(['text']),ParseBudget.seconds(0),1,1)

    async def test_renderer_failure_never_calls_provider_or_raw_pdf_fallback(self):
        class Provider:
            _api_key = 'synthetic'
            model_name = 'synthetic'
            _base_url = 'https://mock.invalid/'
            def __init__(self): self.calls = 0
            async def convert_to_markdown(self,*args,**kwargs):
                self.calls += 1
                raise AssertionError('Must not reach provider')
        provider = Provider()
        from kitng_doc_parser.pdf.chunking import inspect_pdf_pages
        async def broken(operation,data,budget,start=0,end=0):
            if operation == 'inspect': return inspect_pdf_pages(data)
            self.assertEqual(operation,'render_png')
            raise PdfRenderError('renderer_failure')
        with patch('kitng_doc_parser.pipeline.pdf_operation',broken):
            result = await DocumentParser(provider).parse(DocumentRef('s','mixed.pdf','mixed.pdf'),synthetic_pdf(['text','scan']),budget=ParseBudget.seconds(10))
        self.assertEqual((result.status,result.reason_code,result.retryable,result.native_pages),('partial','renderer_failure',False,1))
        self.assertEqual(provider.calls,0)

    async def test_cancel_renderer_cleans_up_child(self):
        before = {process.pid for process in multiprocessing.active_children()}
        task = asyncio.create_task(pdf_operation('render_png',synthetic_pdf(['text']),ParseBudget.seconds(30),1,1))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await asyncio.wait_for(task,5)
        self.assertEqual({process.pid for process in multiprocessing.active_children()} - before,set())

    async def test_original_page_number_and_png_are_used_in_mixed_pipeline(self):
        class Provider:
            _api_key = 'synthetic'
            model_name = 'synthetic'
            _base_url = 'https://mock.invalid/'
            def __init__(self): self.calls = []
            async def convert_to_markdown(self,data,mime,prompt=None,**kwargs):
                self.calls.append((mime,prompt))
                self_outer.assertTrue(data.startswith(b'\x89PNG'))
                return TEXT
        self_outer = self
        provider = Provider()
        result = await DocumentParser(provider).parse(DocumentRef('s','mixed.pdf','mixed.pdf'),synthetic_pdf(['text','scan','text']),budget=ParseBudget.seconds(10))
        self.assertEqual((result.status,result.origin,result.native_pages,result.ocr_pages),('complete','mixed_pdf',2,1))
        self.assertEqual(provider.calls[0][0],'image/png')
        self.assertIn('страницу 2 из 3',provider.calls[0][1])
