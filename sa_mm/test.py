# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 00:59:17 2020

@author: Admin
"""
import unittest
from func import add_function

class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(add_function(1, 3), 4)
        
if __name__ == '__main__':
    unittest.main()