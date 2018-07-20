from configs import cfg
import time
import os

class RecordLog(object):
    def __init__(self, write_to_file_interval = 20, file_name = 'log.txt'):
        self.write_to_file_interval = write_to_file_interval
        self.wait_num_to_file = self.write_to_file_interval
        build_time = '-'.join(time.asctime(time.localtime(time.time())).strip().split(' ')[1:-1])
        build_time = '-'.join(build_time.split(':'))
        log_file_name = build_time

        self.path = os.path.join(cfg.log_dir or cfg.standby_log_dir, log_file_name + "_" + file_name)

        self.storage = []

    def add(self, content = '-'*30, if_time = False, if_print = True, if_save = True):
        time_str = "   --- "+time.asctime(time.localtime(time.time())) if if_time else ''
        log_content = content + time_str
        if if_print:
            print(log_content)
        # check save
        if if_save:
            self.storage.append(log_content)
            self.wait_num_to_file -= 1
            if self.wait_num_to_file == 0:
                self.wait_num_to_file = self.write_to_file_interval
                self.write_to_file()
                self.storage = []

    def write_to_file(self):
        with open(self.path, 'a', encoding = 'utf-8') as file:
            for element in self.storage:
                file.write(element + os.linesep)

    def done(self):
        self.add('Done')

_logger = RecordLog(20)