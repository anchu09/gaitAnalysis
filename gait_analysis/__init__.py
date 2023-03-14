import logging

# Prevent "No handler found" warnings when the library is used without
# logging configured by the calling application (PEP 3151 / logging HOWTO).
logging.getLogger(__name__).addHandler(logging.NullHandler())
