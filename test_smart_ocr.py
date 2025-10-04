#!/usr/bin/env python3
"""
Simple test script to verify smart OCR functionality.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.smart_ocr import smart_ocr_processor
from ingestion.loaders.smart_pdf_loader import SmartPDFLoader
from ingestion.loaders.smart_image_loader import SmartImageLoader


def test_smart_ocr_imports():
    """Test that all smart OCR components can be imported."""
    print("🧪 Testing Smart OCR Imports...")
    
    # Test SmartOCRProcessor
    assert smart_ocr_processor is not None
    print("✅ SmartOCRProcessor imported successfully")
    
    # Test loaders
    pdf_loader = SmartPDFLoader()
    image_loader = SmartImageLoader()
    
    assert pdf_loader is not None
    assert image_loader is not None
    print("✅ Smart loaders initialized successfully")
    
    return True


def test_text_quality_analysis():
    """Test text quality analysis functionality."""
    print("\n🧪 Testing Text Quality Analysis...")
    
    # Test with good quality text
    good_text = "This is a well-formatted document with proper text structure. It contains multiple sentences and paragraphs with good readability."
    analysis = smart_ocr_processor._analyze_text_quality(good_text)
    
    assert analysis["is_readable"] == True
    assert analysis["quality_score"] > 0.5
    print(f"✅ Good text detected as readable (score: {analysis['quality_score']:.2f})")
    
    # Test with poor quality text (OCR artifacts)
    poor_text = "Th1s 1s p00r qu4l1ty t3xt w1th    l0ts 0f   sp4c3s   4nd   fragm3nt3d   w0rds"
    analysis = smart_ocr_processor._analyze_text_quality(poor_text)
    
    assert analysis["is_readable"] == False
    assert analysis["quality_score"] < 0.5
    print(f"✅ Poor text detected as needing OCR (score: {analysis['quality_score']:.2f})")
    
    return True


def test_configuration():
    """Test that smart OCR configuration is properly set."""
    print("\n🧪 Testing Smart OCR Configuration...")
    
    # Check key configuration values
    assert smart_ocr_processor.MIN_TEXT_RATIO > 0
    assert smart_ocr_processor.MAX_WHITESPACE_RATIO < 1
    assert smart_ocr_processor.IMAGE_DPI > 0
    
    print("✅ Smart OCR configuration is valid")
    
    return True


def run_all_tests():
    """Run all smart OCR tests."""
    print("🚀 Starting Smart OCR Tests...\n")
    
    try:
        # Run individual tests
        test_smart_ocr_imports()
        test_text_quality_analysis()
        test_configuration()
        
        print("\n🎉 All Smart OCR tests passed!")
        print("\n📋 Summary:")
        print("   ✅ Smart OCR components imported successfully")
        print("   ✅ Text quality analysis working correctly")
        print("   ✅ Configuration is valid")
        print("\n🎯 Smart OCR system is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)