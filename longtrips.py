#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matsim
import pandas as pd
from collections import defaultdict
from typing import List, Union, Tuple, Dict

folder = 'vdf'

trips_path = f'/home/leonefamily/csfm/vdf_vs_normal/{folder}/output_trips.csv.gz'
population_path = f'/home/leonefamily/csfm/vdf_vs_normal/{folder}/output_plans.xml.gz'
network_path = f'/home/leonefamily/csfm/vdf_vs_normal/{folder}/output_network.xml.gz'

trips = pd.read_csv(trips_path, sep=';')

car_trips = trips[trips['modes'] == 'car']
car_trips['trav_time'] = pd.to_timedelta(
    car_trips['trav_time']
).dt.total_seconds()

long_car_trips = car_trips[car_trips['trav_time'] >= 7200]

affected_ids = {
    str(persid): persdf['trip_number'].tolist()
    for persid, persdf in long_car_trips.groupby('person')
}

reader = matsim.plan_reader(population_path, selected_plans_only=True)
counts = defaultdict(int)

for person, plan in reader:
    # acts = plan.findall('activity')
    legs = plan.findall('leg')
    pid = person.attrib['id']

    if pid not in affected_ids:
        continue

    for i, leg in enumerate(legs, start=1):
        if i not in affected_ids[pid]:
            continue
        links = leg.find('route').text
        if links and leg.attrib['mode'] == 'car':
            for link in links.split():
                counts[link] += 1

counts_df = pd.Series(counts, name='count').to_frame()
counts_df.index.name = 'link_id'
counts_df.reset_index(inplace=True)

net = matsim.read_network(network_path).as_geo('EPSG:2056')
net_counts = net.merge(counts_df)

net_counts.to_file(
    f'/home/leonefamily/csfm/vdf_vs_normal/{folder}/analysis/long_trips.shp',
    encoding='utf-8'
)

long_car_trips.to_csv(
    f'/home/leonefamily/csfm/vdf_vs_normal/{folder}/analysis/long_trips.csv',
    index=False,
    sep=';',
    decimal=','
)