from collections import defaultdict
import os
import re
import numpy as np
import utils


def rel_map(triplets: list):
	"""
	remap equivalent relations lowercase and plural/singular to their most popular variant
	"""
	rels = utils.rel_count(triplets)
	lower_rels = defaultdict(list)
	for rel, count in rels.items():
		lower_rels[rel.lower()].append((rel, count))
	lower_rels_cp = list(lower_rels.keys())
	for lrel in lower_rels_cp:
		if lrel.endswith("ies"):
			lrel_sg = lrel[:-3] + "y"
			if lrel_sg in lower_rels:
				lower_rels[lrel_sg] += lower_rels[lrel]
				del lower_rels[lrel]
		elif lrel.endswith("s"):
			lrel_sg = lrel[:-1]
			if lrel_sg in lower_rels:
				lower_rels[lrel_sg] += lower_rels[lrel]
				del lower_rels[lrel]
	r_map = {}
	for lrel, rels in lower_rels.items():
		if len(rels) > 1:
			max_rel = max(rels, key=lambda rc: rc[1])[0]
			for rel, _ in rels:
				if rel != max_rel:
					r_map[rel] = max_rel
	for h, r, t in triplets:
		yield h, r_map[r] if r in r_map else r, t


def filter_rel(triplets, filter_r):
	for h, r, t in triplets:
		if r not in filter_r:
			yield h, r, t


def space_caps(triplets):
	for h, r, t in triplets:
		yield h, "_".join(re.findall(r"[A-Z]?[^A-Z]+", r)).lower(), t


def sub_g(triplets: list, seeds: set, dist: int):
	seen = [False]*len(triplets)
	prev_set = seeds
	for _ in range(dist):
		next_set = set()
		for i, (h, r, t) in enumerate(triplets):
			if not seen[i] and (h in prev_set or t in prev_set):
				yield h, r, t
				if h in prev_set:
					next_set.add(t)
				else:
					next_set.add(h)
				seen[i] = True
		prev_set = next_set


def split_data(triplets: list, split=.01):
	g = defaultdict(list)
	for h, r, t in triplets:
		g[(h, t)].append(r)
	total_test = int(len(triplets)*split)
	test_count = 0
	valid_count = 0
	train, valid, test = [], [], []
	for (h, t), rels in g.items():
		if len(rels) == 1 and test_count < total_test:
			test.append((h, rels[0], t))
			test_count += 1
		elif len(rels) == 1 and valid_count < total_test:
			valid.append((h, rels[0], t))
			valid_count += 1
		else:
			for r in rels:
				train.append((h, r, t))
	return train, valid, test


def prune_test(train, test, max_dist=4, ratio_off=.1):
	distances = utils.distances_test(train, test, max_dist)
	total_off = np.sum(distances > max_dist)
	target_off = (len(test)-total_off)*ratio_off/(1-ratio_off)
	count_off = 0
	for (h, r, t), dist in zip(test, distances):
		if dist <= max_dist:
			yield h, r, t
		elif count_off < target_off:
			yield h, r, t
			count_off += 1


def make_data_set(path, filter_r=None, seed=None, dist=3, split=.01):
	with open(path, "r") as fdata:
		lines = list(fdata)
	directory, filename = os.path.split(path)
	# we remove the file extension if there is one
	filename = filename.rsplit(".")[0]
	print(f"{path} {len(lines):,}")
	# Remap the relations
	triplets = list(utils.it_triplets(lines))
	triplets = rel_map(triplets)
	if filter_r:
		# Filter the relations
		triplets = list(filter_rel(triplets, filter_r))
	# Relabel the relations
	triplets = list(space_caps(triplets))
	if filter_r:
		utils.save_triplets(triplets, os.path.join(directory, f"{filename}_clean.ttl"))
	if seed:
		# Subset based on distance from seed
		triplets = list(sub_g(triplets, seed, dist))
		utils.save_triplets(triplets, os.path.join(directory, f"{filename}_sub.ttl"))
	# Split dataset into train, valid, test
	train, valid, test = split_data(triplets, split)
	utils.save_triplets(train, os.path.join(directory, f"{filename}_train.ttl"))
	# Prune test dataset based on distance in test
	valid = prune_test(train, valid, dist)
	utils.save_triplets(valid, os.path.join(directory, f"{filename}_valid.ttl"))
	test = prune_test(train, test, dist)
	utils.save_triplets(test, os.path.join(directory, f"{filename}_test.ttl"))


if __name__ == '__main__':
	make_data_set("WikiData/infobox_en.ttl", seed={"Barack_Obama", "Donald_Trump"})
