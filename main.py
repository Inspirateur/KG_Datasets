from collections import defaultdict
import sys
from random import shuffle
from tqdm import tqdm, trange
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra


def load_dataset():
	with open("Datasets/Wiki/infobox_sub_train.ttl", "r") as ftrain:
		train = [line.strip().split() for line in ftrain]
	with open("Datasets/Wiki/infobox_sub_test_trim.ttl", "r") as ftest:
		test = [line.strip().split() for line in ftest]
	return train, test, test


def sub_wiki(seeds: set, dist: int):
	lines = []
	prev_set = seeds
	for i in trange(dist, desc="Sub Wiki", file=sys.stdout, ncols=140):
		next_set = set()
		with open("Datasets/Wiki/infobox_en_clean.ttl", "r") as fdata:
			for line in tqdm(fdata, desc=f"Pass {i+1}", total=14_834_878, file=sys.stdout, ncols=140, leave=False):
				h, r, t = line.strip().split()
				if h in prev_set or t in prev_set:
					lines.append(line)
					next_set.add(h)
					next_set.add(t)
		prev_set = next_set
	shuffle(lines)
	print(f"Finished sub wiki ({len(lines):,} lines)")
	with open("Datasets/Wiki/infobox_sub.ttl", "w") as fsub:
		fsub.writelines(lines)


def split_wiki(split=.002):
	g = defaultdict(list)
	total = 13_589_039
	with open("Datasets/Wiki/infobox_sub.ttl", "r") as fdata:
		for line in tqdm(fdata, desc=f"Wikiboxes", total=total, file=sys.stdout, ncols=140):
			h, r, t = line.strip().split()
			g[(h, t)].append(r)
	total_test = int(total*split)
	test_count = 0
	with open("Datasets/Wiki/infobox_sub_test.ttl", "w") as ftest:
		with open("Datasets/Wiki/infobox_sub_train.ttl", "w") as ftrain:
			for (h, t), rels in tqdm(g.items(), desc="Split Wiki", file=sys.stdout, ncols=140):
				if len(rels) == 1 and test_count < total_test:
					ftest.write(" ".join((h, rels[0], t)) + "\n")
					test_count += 1
				else:
					for r in rels:
						ftrain.write(" ".join((h, r, t)) + "\n")


def test_distances(limit) -> np.ndarray:
	train, valid, test = load_dataset()
	# map entities to an id
	emap = {}
	for h, _, t in train:
		if h not in emap:
			emap[h] = len(emap)
		if t not in emap:
			emap[t] = len(emap)

	# build the kg
	kg = lil_matrix((len(emap), len(emap)), dtype=np.uint16)
	for h, _, t in train:
		kg[emap[h], emap[t]] = 1
	kg = kg.tocsr()
	idx, _ = zip(*sorted(enumerate(test), key=lambda i_hrt: i_hrt[1][0]))
	distances = [0]*len(idx)
	_h = None
	shortest = None
	for i in tqdm(idx, desc="Distances", ncols=140):
		h, r, t = test[i]
		if _h != h:
			shortest = dijkstra(
				kg, limit=limit, indices=emap[h], return_predecessors=False
			)
			_h = h
		distances[i] = shortest[emap[t]]
	distances = np.array(distances)
	distances[distances > limit] = limit + 1
	return distances


def prune_test_wiki(max_dist=4, ratio_off=.1):
	distances = test_distances(max_dist)
	with open("Datasets/Wiki/infobox_sub_test.ttl", "r") as ftest:
		test = list(ftest)
	total = len(test)
	total_off = np.sum(distances > max_dist)
	target_off = (total-total_off)*ratio_off/(1-ratio_off)
	count_off = 0
	with open("Datasets/Wiki/infobox_sub_test_trim.ttl", "w") as ftesttrim:
		for line, dist in zip(test, distances):
			if dist <= max_dist:
				ftesttrim.write(line)
			elif count_off < target_off:
				ftesttrim.write(line)
				count_off += 1
