from collections import defaultdict
try:
	# very important to generate negative examples quickly
	import Pyewacket as random
except ModuleNotFoundError:
	import random
import sys
from tqdm import tqdm
import utils


def depth_amount(max_depth, n):
	"""
	Returns a list of tuple: depth, amount of path>=5
	Make it so that depth u_n+1 is 2 times depth u_n,
	using a geometric serie formula. Because as path depth grows,
	exponentially more paths are needed to cover a fraction of possible paths.
	"""
	q = 3
	u = (1-q)/(1-q**(max_depth-1))
	for d in range(2, max_depth+1):
		yield d, max(int(u*n), 5)
		u *= q


def sample_safe(population, k):
	if len(population) >= k:
		return random.sample(population, k)
	return population + random.choices(population, k=k-len(population))


def generate(path, max_depth=4, n=5):
	"""
	Generate a file with negative examples for each train, valid, test file.
	Each line contains n negative target associated with the head and relation of the same line in the
	corresponding train/valid/test file.
	:param path: path that will be completed with _[train|valid|test].ttl to get the
	train, valid, test files
	:param max_depth: the maximum distance between bad target and head in training set
	:param n: the number of negative example per positive example to generate
	"""
	# <r, {t, }>
	r_t = defaultdict(set)
	# <h, [(r, t), ]>
	g = defaultdict(list)
	with open(f"{path}_train.ttl", "r") as ftrain:
		lines = list(ftrain)
	for h, r, t in utils.it_triplets(lines):
		g[h].append((r, t))
		r_t[r].add(t)
		g[t].append((f"-{r}", h))
	entities = list(g.keys())

	def bad_targets(h, r) -> list:
		nonlocal n
		assert max_depth >= 2
		# if the relation is unknown, sample random entities
		if not r_t[r]:
			return random.sample(entities, n)
		if h not in g:
			return sample_safe(list(r_t[r]), n)
		neg_t = r_t[r] - {n for r, n in g[h] if n in r_t[r]}
		res = set()
		for depth, amount in depth_amount(max_depth, 1_000):
			for i in range(amount):
				traversed = set()
				node = h
				for d in range(depth):
					_, neigh = random.choice(g[node])
					if neigh in traversed:
						break
					if neigh in neg_t:
						res.add(neigh)
						if len(res) >= n:
							return list(res)
						break
					traversed.add(neigh)
		# we complete with k negative target that might exeed max_depth
		k = n - len(res)
		neg_t -= res
		if neg_t:
			return list(res) + sample_safe(list(neg_t), k)
		return sample_safe(list(r_t[r]), n)

	splits = ["train", "valid", "test"]
	for split in splits:
		with open(f"{path}_{split}.ttl", "r") as fsplit:
			lines = list(fsplit)
		with open(f"{path}_neg_{split}.ttl", "w") as fnegsplit:
			triplets = list(utils.it_triplets(lines))
			for h, r, _ in tqdm(triplets, desc=f"{split} neg", ncols=140, file=sys.stdout):
				fnegsplit.write(" ".join(bad_targets(h, r))+"\n")


if __name__ == '__main__':
	generate("WikiData/infobox_en")
