import logging
import os
import warnings
from collections import namedtuple
import pprint
from typing import Union
import matplotlib.pyplot as plt

SFLogger = namedtuple("SFLogger", ["logger", "file_handler", "console_handler"])
LOGFILE = 'log.txt'
global al_logger
#al_logger: Union[SFLogger, None] = None
al_logger = None



# set up pretty printer
# pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)
pp = pprint.PrettyPrinter(indent=2,)

def log_pretty(obj):
    pretty_out = f"{pp.pformat(obj)}"

    return f'{pretty_out}\n'

def load_existing_logger():
    logger = logging.Logger.manager.loggerDict.get("sal", None)
    if logger is None:
        return None
    if isinstance(logger.handlers[0],logging.FileHandler):
        fh = logger.handlers[0]
        ch = logger.handlers[1]
    else:
        ch = logger.handlers[0]
        fh = None
    al_logger = SFLogger(logger, fh, ch)
    return al_logger


def init_global_logger(root_path, experiment_path, use_fh=False, print_logger=False):
    global al_logger
    if al_logger is not None:
        warnings.warn("Logger already initialized")
    if print_logger:
        al_logger = SFLogger(PrintLogger(), None, None)
    else:
        logger = logging.getLogger("p2f")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if use_fh:
            fh = logging.FileHandler(os.path.join(root_path, experiment_path, LOGFILE))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            fh = None
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # add the handlers to the logger

        logger.addHandler(ch)
        al_logger = SFLogger(logger, fh, ch)
        al_logger.logger.info('Logger initialized')
    return al_logger


def deinit_global_logger(save_file=True):
    global al_logger
    if al_logger is None:
        warnings.warn("Logger not initialized")

    al_logger.logger.info('deinitialize Logger')
    if al_logger.file_handler is not None:
        al_logger.file_handler.close()
    if al_logger.console_handler is not None:
        al_logger.console_handler.flush()
    al_logger.logger.handlers.clear()
    del logging.Logger.manager.loggerDict["sal"]
    l,f,c = al_logger.logger, al_logger.file_handler, al_logger.console_handler
    al_logger = None
    del l
    del f
    del c


def get_logger():
    global al_logger
    if al_logger is None:
        al_logger = load_existing_logger()
        if al_logger is None:
            al_logger = init_global_logger(None, None, False, True)
            print("Logger not initialized, initializing print logger")
        else:
            print("Logger is None, but have been loaded from logger manager")
    return al_logger


def gl_info(message):
    get_logger().logger.info(message)


def gl_debug(message):
    get_logger().logger.debug(message)


def gl_warning(message):
    get_logger().logger.warning(message)


def gl_error(message):
    get_logger().logger.error(message)


def gl_critical(message):
    get_logger().logger.critical(message)


class PrintLogger():
    @staticmethod
    def info(message):
        print(message)

    @staticmethod
    def error(message):
        print(message)

    @staticmethod
    def warn(message):
        warnings.warn(message)

    @staticmethod
    def debug(message):
        print(message)

    @staticmethod
    def critical(message):
        print(message)
