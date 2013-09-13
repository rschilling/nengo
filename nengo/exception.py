"""
Exception objects
=================

These classes are raised as exceptions both internally and externally.
"""


class ShapeMismatch(ValueError):
    """Shape sizes are unequal."""

    def __init__(self, *shapes):
        self.shapes = shapes

    def __str__(self):
        return 'Mismatch in given shapes: %s' % ', '.join(self.shapes) 


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""
    pass
