#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import sys
import argparse
import math
import codecs
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda,
                                           beam_trace=opt.dump_beam != "",
                                           min_length=opt.min_length)

    while 1:
	line = sys.stdin.readline().strip()
	flist = line.split("\t")
	if len(flist) != 2:
	    continue

        opt.src = "./src.tmp"
	fp_src = open(opt.src, "w")
	fp_src.write(flist[0] + "\n")
	fp_src.close()

	opt.tgt = "./tgt.tmp"
	fp_tgt = open(opt.tgt, "w")
	fp_tgt.write(flist[1] + "\n")
	fp_tgt.close()
	
        # Test data
        data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        data_iter = onmt.io.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, translator.fields,
            opt.n_best, opt.replace_unk, opt.tgt)

        # Translator
        # Statistics
        counter = count(1)

        for batch in data_iter:
	    tag_list = []
	    z_true_list = []
	    response_list = []
	    z_mu_list = []; z_logvar_list = []
	    pred_score_list = []; gold_score_list = []
	    context_list = []; gold_truth_list = []

	    # Sample z_true from true_mu_dist, true_logvar_dist
	    for i in range(0, 3):
                batch_data, z_true, true_mu_dist, true_logvar_dist = translator.translate_batch(batch, data)
                translations = builder.from_batch(batch_data)

		pred_score_tmp = []; gold_score_tmp = []
		response_list_tmp = []
                for trans in translations:
                    pred_score_tmp.append(trans.pred_scores[0])
		    if i == 0:
                        gold_score_list.append(trans.gold_score) 

                    n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:opt.n_best]]
		    response_list_tmp.append(n_best_preds[0])

	        if i == 0:
		    z_mu_list = true_mu_dist
		    z_logvar_list = true_logvar_dist
                    context_list.append(" ".join(trans.src_raw))
		    gold_truth_list.append(" ".join(trans.gold_sent))

		z_true_list.append(z_true)
		pred_score_list.append(pred_score_tmp)
		response_list.append(response_list_tmp)
		tag_list.append("Sample from distribution-" + str(i))

	    # Sample z_true randomly
	    for i in range(0, 3):
                batch_data, z_true, true_mu_dist, true_logvar_dist = translator.translate_batch(batch, data, tag="random")
                translations = builder.from_batch(batch_data)

		pred_score_tmp = []; gold_score_tmp = []
		response_list_tmp = []
                for trans in translations:
                    pred_score_tmp.append(trans.pred_scores[0])
                    gold_score_tmp.append(trans.gold_score)

                    n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:opt.n_best]]
		    response_list_tmp.append(n_best_preds[0])

		z_true_list.append(z_true)
		pred_score_list.append(pred_score_tmp)
		gold_score_list.append(gold_score_tmp)
		response_list.append(response_list_tmp)
		tag_list.append("Sample randomly-" + str(i))

	    for i in range(0, len(context_list)):
		print "CONTEXT: ", context_list[i]
		print "GOLD: ", gold_truth_list[i]
		print "GOLD SCORE: ", gold_score_list[i]
		print "Z-MU: ", [item.view(-1).data.cpu().numpy() for item in z_mu_list[i]]
		print "Z-LOGVAR: ", [item.view(-1).data.cpu().numpy() for item in z_logvar_list[i]]
		print "=================================================="
		for j in range(0, len(tag_list)):
		    print tag_list[j]
		    print "PRED: ", response_list[j][i]
		    print "PRED SCORE: ", pred_score_list[j][i]
		    print "Z: ", [item.view(-1).data.cpu().numpy() for item in z_true_list[j][i]]
		    print ""

        if opt.dump_beam:
            import json
            json.dump(translator.beam_accum, codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
