#coding=utf8

import sys
import random

SRC_PATH = "../data/"
TGT_PATH = "./data/"

def get_lines(domain):
	en_list = []; vi_list = []; lines = []
	for line in open(SRC_PATH + "/" + domain + ".en"):
		en_list.append(line.strip())
	for line in open(SRC_PATH + "/" + domain + ".vi"):
		vi_list.append(line.strip())
	for i in range(0, len(en_list)):
		lines.append([en_list[i], vi_list[i]])
	return lines

def write_back(lines, domain):
	en_fpout = open(TGT_PATH + "/" + domain + ".en", "w")
	vi_fpout = open(TGT_PATH + "/" + domain + ".vi", "w")
	discriminator_fpout = open(TGT_PATH + "/" + domain + ".disc", "w")
	for item in lines:
		en_fpout.write(item[0] + "\n")
		vi_fpout.write(item[1] + "\n")
		discriminator_fpout.write(item[0] + " <s> " + item[1] + "\t1\n")

	neg_num = int(len(lines) * 0.5)
	neg_ctx = random.sample(lines, neg_num)
	neg_res = random.sample(lines, neg_num)
	for i in range(0, neg_num):
	    en_fpout.write(neg_ctx[i][0] + "\n")
	    vi_fpout.write(neg_ctx[i][1] + "\n")
	    discriminator_fpout.write(neg_ctx[i][0] + " <s> " + neg_ctx[i][1] + "\t0\n")
	en_fpout.close()
	vi_fpout.close()
	discriminator_fpout.close()

if __name__ == '__main__':
	lines_train = get_lines("train")
	lines_test = get_lines("test")
	lines_dev = get_lines("dev")

	write_back(lines_train, "train")
	write_back(lines_test, "test")
	write_back(lines_dev, "dev")
