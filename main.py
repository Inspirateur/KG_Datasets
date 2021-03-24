from collections import defaultdict
import os
import re
from tqdm import tqdm
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra


def rel_count(triplets):
	rels = defaultdict(int)
	for _, r, _ in triplets:
		rels[r] += 1
	return rels


def rel_map(triplets: list):
	"""
	remap equivalent relations lowercase and plural/singular to their most popular variant
	"""
	rels = rel_count(triplets)
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
		if r in filter_r:
			yield h, r, t


def space_caps(triplets):
	for h, r, t in triplets:
		yield h, "_".join(re.findall(r"[A-Z]?[^A-Z]+", r)), t


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


def prune_test(train, test, max_dist=4, ratio_off=.1):
	distances = distances_test(train, test, max_dist)
	total_off = np.sum(distances > max_dist)
	target_off = (len(test)-total_off)*ratio_off/(1-ratio_off)
	count_off = 0
	for (h, r, t), dist in zip(test, distances):
		if dist <= max_dist:
			yield h, r, t
		elif count_off < target_off:
			yield h, r, t
			count_off += 1


def it_triplets(lines):
	for line in lines:
		yield line.strip().split()


def save_triplets(triplets, fname):
	count = 0
	with open(fname, "w") as fsave:
		for h, r, t in triplets:
			fsave.write(" ".join((h, r, t))+"\n")
			count += 1
	print(fname, f"{count:,}")


def make_data_set(path, filter_r=None, seed=None, dist=4, split=.01):
	with open(path, "r") as fdata:
		lines = list(fdata)
	directory, filename = os.path.split(path)
	# we remove the file extension if there is one
	filename = filename.rsplit(".")[0]
	print(f"{path} {len(lines):,}")
	# Remap the relations
	triplets = list(it_triplets(lines))
	triplets = rel_map(triplets)
	if filter_r:
		# Filter the relations
		triplets = list(filter_rel(triplets, filter_r))
	# Relabel the relations
	triplets = list(space_caps(triplets))
	save_triplets(triplets, os.path.join(directory, f"{filename}_clean.ttl"))
	if seed:
		# Subset based on distance from seed
		triplets = list(sub_g(triplets, seed, dist))
		save_triplets(triplets, os.path.join(directory, f"{filename}_sub.ttl"))
	# Split dataset into train, valid, test
	train, valid, test = split_data(triplets, split)
	save_triplets(train, os.path.join(directory, f"{filename}_train.ttl"))
	# Prune test dataset based on distance in test
	valid = prune_test(train, valid, dist)
	save_triplets(valid, os.path.join(directory, f"{filename}_valid.ttl"))
	test = prune_test(train, test, dist)
	save_triplets(test, os.path.join(directory, f"{filename}_test.ttl"))


if __name__ == '__main__':
	"""
	_filter_r = {
		"subdivisionType", "subdivisionName", "birthPlace", "location", "deathPlace", "title", "type", "label",
		"almaMater", "name", "country", "position", "predecessor", "writer", "successor", "city", "occupation",
		"producer", "nationality", "language", "director", "battles", "associatedActs", "youthclubs", "awards",
		"residence", "sport", "office", "industry", "publisher", "party", "origin", "education", "format", "region",
		"distributor", "headquarters", "state", "college", "owner", "religion", "field", "platforms", "spouse",
		"venue", "workplaces", "column", "counties", "operator", "locationCity", "modes", "branch"
		"president", "markTitle", "products", "studio", "manufacturer", "network", "affiliations", "address",
		"knownFor", "primeminister", "fields", "place", "locationCountry", "ground", "candidate", "area", "leaderName",
		"seat", "ideology", "company", "nearestCity", "parent", "rank", "discipline", "profession", "governmentType",
		"district", "founder", "borough", "highSchool", "areaServed", "influences", "commands", "relatives", "states",
		"county", "range", "owned", "conference", "preceded", "succeeded", "appointer", "authority", "allegiance",
		"province", "cityServed", "event", "religiousAffiliation", "subject", "doctoralAdvisor", "governingBody",
		"children", "leader", "parents", "channel", "monarch", "presenter", "church", "pastMembers", "affiliation",
		"school", "keyPeople", "commander", "workInstitutions", "seatType", "movement", "restingPlace", "governor",
		"citizenship", "employer", "nationalOrigin", "campus", "father", "hqLocationCity", "workInstitution", "influenced",
		"relations", "mainInterests", "license", "hometown", "department", "doctoralStudents", "order", "house", "tenants",
		"cities", "veneratedIn", "countryAdminDivisions", "status", "constituency", "placeofburial", "wars", "precededBy",
		"source", "followedBy", "assembly", "category", "worldPlace", "executiveProducer", "university", "minister",
		"officeholder", "judges", "partof", "garrison", "deputy", "namedFor", "jurisdiction", "builder", "family",
		"hqLocationCountry", "bishop", "launchSite", "locationTown", "rels", "homeTown", "birthplace", "ethnicity",
		"otherparty", "commonLanguages", "combatant", "architecturalStyle", "schoolTradition", "owners"
	}
	make_data_set("WikiData/infobox_en.ttl", _filter_r, {"Barack_Obama", "Donald_Trump"})
	"""
	with open("WikiData/infobox_en_train.ttl", "r") as ftrain:
		print(len(list(ftrain)))
