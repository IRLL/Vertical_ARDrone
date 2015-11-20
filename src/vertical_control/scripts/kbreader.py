from __future__ import print_function
import readchar
import os
import signal


class KBreader():
	def __init__(self):
		signal.signal(signal.SIGALRM, self.handler)

	def handler(self, signum, frame):
		raise Exception("keyboard timeout triggered")

	def test_quit(self, key):
		if key == chr(3):
			os.kill(os.getpid(), signal.SIGINT)

	def getkey(self, timeout = 1):
		signal.alarm(timeout)
		try:
			key = readchar.readkey()
		except Exception, execpt:
			key = []


		self.test_quit(key)
		return key


if __name__ == '__main__':
	kr = KBreader()

	while 1:
		key = kr.getkey()
		tmp = []
		for ch in key:
			tmp.append(ord(ch))
		print("got key: ", tmp)
