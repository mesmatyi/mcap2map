#!/usr/bin/env python3
"""Routing neighbourhood of lanelets: following / previous / lane-change neighbours, end points.
usage: link_diag.py MAP_OSM ID [ID ...]"""
import os, sys
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LAT0, LON0
m, _ = lanelet2.io.loadRobust(sys.argv[1], UtmProjector(Origin(LAT0, LON0)))
rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
g = lanelet2.routing.RoutingGraph(m, rules)
for a in map(int, sys.argv[2:]):
    ll = m.laneletLayer[a]
    L, Rb = ll.leftBound, ll.rightBound
    f = lambda p: f"({p.x:.1f},{p.y:.1f})"
    lt = lambda ls: (ls.attributes["type"] if "type" in ls.attributes else "") + "/" + (ls.attributes["subtype"] if "subtype" in ls.attributes else "")
    print(f"{a}: {ll.attributes['subtype']} turn={ll.attributes['turn_direction'] if 'turn_direction' in ll.attributes else '-'} "
          f"start L{f(L[0])} R{f(Rb[0])} end L{f(L[-1])} R{f(Rb[-1])} | left bound {lt(L)} right bound {lt(Rb)}")
    print(f"   following {[x.id for x in g.following(ll)]} previous {[x.id for x in g.previous(ll)]} "
          f"left {g.left(ll).id if g.left(ll) else None} right {g.right(ll).id if g.right(ll) else None} "
          f"adjLeft {g.adjacentLeft(ll).id if g.adjacentLeft(ll) else None} adjRight {g.adjacentRight(ll).id if g.adjacentRight(ll) else None}")
