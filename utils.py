from collections import defaultdict
import functools
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm


def it_triplets(lines):
	for line in lines:
		yield line.strip().split()


def rel_count(triplets):
	rels = defaultdict(int)
	for _, r, _ in triplets:
		rels[r] += 1
	return rels


def save_triplets(triplets, fname):
	count = 0
	with open(fname, "w") as fsave:
		for h, r, t in triplets:
			fsave.write(" ".join((h, r, t))+"\n")
			count += 1
	print(fname, f"{count:,}")


def distances_test(train, test, limit) -> np.ndarray:
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
		if h not in emap or t not in emap:
			distances[i] = limit + 1
			continue
		if _h != h:
			shortest = dijkstra(
				kg, limit=limit, indices=emap[h], return_predecessors=False
			)
			_h = h
		distances[i] = shortest[emap[t]]
	distances = np.array(distances)
	distances[distances > limit] = limit + 1
	return distances


# relations in FreeBase are overly long, we use this to shorten them
@functools.lru_cache()
def shorten_r(rel):
	rs = rel.split(".")
	res = []
	# get a context
	parts = list(filter(None, rs[0].split("/")))
	ctx = None
	for ctx in parts[:-1]:
		if ctx != parts[-1]:
			break
	for r in rs:
		last_part = r.rsplit("/", 1)[-1]
		tokens = [t for t in last_part.split("_") if t != ctx]
		if tokens:
			res.append("_".join(tokens))
		else:
			res.append(last_part)
	ctx = ctx+"/" if ctx is not None else ""
	return ctx + ".".join(res)
