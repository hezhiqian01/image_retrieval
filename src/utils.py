"""Control log function."""

import os.path
import logging
import logging.handlers
import sys
import threading


def get_logger():
    log = MyLog()
    log.start("run", "./logs")
    return log.logger


class MyLog(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(MyLog, '_instance'):
            with MyLog._instance_lock:
                if not hasattr(MyLog, "_instance"):
                    MyLog._instance = object.__new__(cls)
        return MyLog._instance

    def __init__(self):
        if not hasattr(self, 'logger'):
            self.logger = self._get_default_logger()

    def _get_default_logger(self):
        try:
            name = self.__class__.__name__
            logger = logging.getLogger(name)
            logger.propagate = 0
            hl = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s][%(thread)d][%(threadName)s][%(levelname)s]%(message)s[file:%(filename)s,func:%(funcName)s,line:%(lineno)d]")
            hl.setFormatter(formatter)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                logger.addHandler(hl)
            return logger
        except Exception as e:
            print(e)
            return None

    def start(self, filename, logpath, when="midnight", interval=1, backupCount=0):
        """Start the log module."""
        self.__logfilename = filename
        self.__logpath = logpath

        # 没有日志存储路径的文件夹，创建一个文件夹
        if not os.path.exists(self.__logpath):
            os.mkdir(self.__logpath)

        self.logger = logging.getLogger(self.__logfilename)
        formatter = logging.Formatter(
            "[%(asctime)s][%(thread)d][%(threadName)s][%(levelname)s]%(message)s[file:%(filename)s,func:%(funcName)s,line:%(lineno)d]")
        self.logger.setLevel(logging.INFO)
        filepathname = os.path.join(self.__logpath, self.__logfilename + ".log")

        # disable parent logger
        self.logger.propagate = 0
        try:
            fh = logging.handlers.TimedRotatingFileHandler(filepathname, when=when, interval=interval,
                                                           backupCount=backupCount)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(fh)
        except Exception as e:
            print("haha, error", e)

        return self.logger


def __test():
    my_log = MyLog()
    my_log.logger.info('output console.')
    my_log.start("run", "../logs")
    my_log.logger.info("asdsad")
    print("-------")
    my_log.logger.info("adasdas")
    my_log.logger.error("adasdas")
    print("-------")


if __name__ == "__main__":
    __test()
