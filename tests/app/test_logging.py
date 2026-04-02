import unittest
import logging
import os
from src.app.logging import setup_logging

class TestLogging(unittest.TestCase):
    def test_setup_logging(self):
        log_file = 'test.log'
        setup_logging(log_file=log_file, level=logging.DEBUG)
        
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Test that the file handler was added
        self.assertTrue(any(isinstance(h, logging.FileHandler) for h in logger.handlers))
        
        # Clean up the log file
        os.remove(log_file)

if __name__ == '__main__':
    unittest.main()
