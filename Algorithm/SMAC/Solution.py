

class Solution:
    def __init__(self, independent, dependent=None):
        self.independent = independent
        self.dependent = dependent

    def add_dependent(self, dependent):
        self.dependent = dependent