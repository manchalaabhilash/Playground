import unittest
import os
import sys
import tempfile
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import DocumentProcessor, ImageProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Create a temporary text file for testing
        self.temp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        self.temp_txt.write(b"This is a test document.\nIt has multiple lines.\nThis is for testing the document processor.")
        self.temp_txt.close()
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.temp_txt.name):
            os.unlink(self.temp_txt.name)
    
    def test_process_documents(self):
        # Test with a text file
        chunks = self.processor.process_documents([self.temp_txt.name])
        
        # Check that chunks were created
        self.assertGreater(len(chunks), 0)
        
        # Check that the content is in the chunks
        all_text = " ".join([chunk.page_content for chunk in chunks])
        self.assertIn("test document", all_text)
        self.assertIn("multiple lines", all_text)
    
    def test_nonexistent_file(self):
        # Test with a nonexistent file
        chunks = self.processor.process_documents(["nonexistent_file.txt"])
        
        # Should return an empty list
        self.assertEqual(len(chunks), 0)

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ImageProcessor()
        
        # Create a temporary image file for testing
        self.temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        self.temp_img.close()
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.temp_img.name)
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.temp_img.name):
            os.unlink(self.temp_img.name)
    
    def test_process_images(self):
        # Test with an image file
        chunks = self.processor.process_images([self.temp_img.name])
        
        # Check that chunks were created
        self.assertEqual(len(chunks), 1)
        
        # Check that the metadata is correct
        self.assertEqual(chunks[0]["image_path"], self.temp_img.name)
        self.assertEqual(chunks[0]["width"], 100)
        self.assertEqual(chunks[0]["height"], 100)
        self.assertEqual(chunks[0]["format"], "JPEG")
    
    def test_nonexistent_file(self):
        # Test with a nonexistent file
        chunks = self.processor.process_images(["nonexistent_image.jpg"])
        
        # Should return an empty list
        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()