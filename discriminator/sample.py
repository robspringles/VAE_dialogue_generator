#coding=utf8

import sys
import random

SRC_PATH = "../data/"
TGT_PATH = "./data/"
disc_patt = "{\"text\": \"<text>\", \"label\": \"<label>\"}"

def sample_positive(domain):
	en_list = []; vi_list = []; lines = []
	for line in open(SRC_PATH + "/" + domain + ".en"):
		en_list.append(line.strip())
	for line in open(SRC_PATH + "/" + domain + ".vi"):
		vi_list.append(line.strip())
	for i in range(0, len(en_list)):
		lines.append([en_list[i], vi_list[i]])

	pos = []; neg = []
	for item in lines:
		rnum = random.uniform(0, 1)
		if rnum < 0.5:
			neg.append(item)
		else:
			pos.append(item)

	return pos, neg

def sample_negtive(pos_data, neg_data, domain):
	en_fpout = open(TGT_PATH + "/" + domain + ".en", "w")
	vi_fpout = open(TGT_PATH + "/" + domain + ".vi", "w")
	discriminator_fpout = open(TGT_PATH + "/" + domain + ".disc", "w")
	for item in pos_data:
		en_fpout.write(item[0] + "\n")
		vi_fpout.write(item[1] + "\n")
		en_fpout.write(item[0] + "\n")
		neg_sample = random.sample(neg_data, 1)[0][1]
		vi_fpout.write(neg_sample + "\n")
		discriminator_fpout.write(item[0] + " <s> " + item[1] + "\t1\n")
		discriminator_fpout.write(item[0] + " <s> " + neg_sample + "\t0\n")
	en_fpout.close()
	vi_fpout.close()
	discriminator_fpout.close()

if __name__ == '__main__':
	pos_train, neg_train = sample_positive("train")
	pos_test, neg_test = sample_positive("test")
	pos_dev, neg_dev = sample_positive("dev")

	sample_negtive(pos_train, pos_train, "train")
	sample_negtive(pos_test, neg_train, "test")
	sample_negtive(pos_dev, neg_train, "dev")
