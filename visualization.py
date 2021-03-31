import matplotlib.pyplot as plt
import cleaning
import utils


def rel_count(path):
	with open(path, "r") as file:
		lines = list(file)
	triplets = list(utils.it_triplets(lines))
	triplets = cleaning.rel_map(triplets)
	rels = sorted(utils.rel_count(triplets).items(), key=lambda kv: kv[1], reverse=True)[:50]
	labels, count = zip(*rels)
	labels = list(map(utils.shorten_r, labels))
	x = list(range(len(count)))
	plt.figure(figsize=(10, 6))
	plt.bar(x, count)
	plt.xticks(x, labels, rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig(fname="rel_count")


if __name__ == '__main__':
	rel_count("FB15k-237/fb_train.ttl")
