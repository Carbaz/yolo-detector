"""Initialization module for the AI YOLO detection application."""

from logging import basicConfig, getLogger


# Setup the global logger.
LOG_STYLE = '{'
LOG_LEVEL = 'INFO'
LOG_FORMAT = ('{asctime} {levelname:<8} {processName}({process}) '
              '{threadName} {name} {lineno} "{message}"')
basicConfig(level=LOG_LEVEL, style='{', format=LOG_FORMAT)

getLogger(__name__).info('INITIALIZED AI YOLO DETECTOR')
