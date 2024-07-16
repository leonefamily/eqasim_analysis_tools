#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:09:10 2024

@author: leonefamily
"""

import matsim
import momepy
import networkx as nx
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Tuple

CRS = 'EPSG:2056'
QUANTILES = [0, .01, .25, .5, .75, .99, 1]


net_path = (
    '/home/leonefamily/csfm/runs/switzerland/'
    'matsim.simulation.prepare__323656762bf1d320c148278807d409a6.cache/'
    'switzerland_network.xml.gz'
)

net = matsim.read_network(net_path)
net_gdf = net.as_geo(CRS)
car_net_gdf = net_gdf[net_gdf['modes'].str.contains('car')]

print('Nodes count:', len(net.nodes))
print('Links count:', len(net.links))
print('Car links count:', len(car_net_gdf))

qs = car_net_gdf['length'].quantile(QUANTILES).round(2)
print('Quantiles of assigned car links length:\n', qs.to_dict(), sep='')

geo_qs = car_net_gdf.length.quantile(QUANTILES).round(2)
print('Quantiles of Euclidean car links length:\n', geo_qs.to_dict(), sep='')

graph = momepy.gdf_to_nx(car_net_gdf)

centrality_graph = momepy.straightness_centrality(graph)
