import math

class ProgressBar:
    def __init__(self, total, length=80, do_carriage_return=True):
        self.total = total
        self.curr = 0
        self.length = length
        self.num_progress = 0
        self.prev_num_progress = 0
        self.curr_line = ""
        self.do_carriage_return = do_carriage_return

    def _print_progress(self, front_msg="", back_msg=""):
        if self.do_carriage_return:
            lines = ["|", "=" * self.num_progress, ">", " " * (self.length - self.num_progress), "|"]
            progress_bar_line = ''.join(lines)
            stat_line = "  (%d/%d)" % (self.curr, self.total)
            self.curr_line = front_msg + progress_bar_line + stat_line + back_msg
            print(self.curr_line, end="")
        else:
            diff = self.num_progress - self.prev_num_progress
            print("=" * diff, end="")
            if self.num_progress == self.length:
                print(">|  ", end="")

    @staticmethod
    def print_total_line(length):
        lines = ["|", "=" * length, ">|  "]
        total_line = ''.join(lines)
        print(total_line, end="")

    def start(self, front_msg="", back_msg=""):
        if self.do_carriage_return:
            self._print_progress(front_msg=front_msg, back_msg=back_msg)
        else:
            print("|", end="")

    def update(self, n=1, front_msg="", back_msg=""):
        self.curr += n
        ratio_complete = self.curr / self.total
        self.num_progress = min(self.length, math.floor(ratio_complete * self.length))
        if self.do_carriage_return:
            print("\r", end="")
        self._print_progress(front_msg=front_msg, back_msg=back_msg)
        self.prev_num_progress = self.num_progress

    def reset(self):
        if self.do_carriage_return:
            print("\r" + " " * len(self.curr_line), end="\r")
        self.curr = 0
        self.curr_line = ""
        self.num_progress = 0
        self.prev_num_progress = 0
