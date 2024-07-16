#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:24:24 2024

@author: leonefamily
"""

import xml
import math
import copy
import matsim
import pickle
import matplotlib
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Union, Dict, Any, Optional, Callable, Literal, List, Tuple


def str_to_bool(
        s: Optional[str]
) -> bool:
    """Get True if string have value of `true`, otherwise False"""
    return s == 'true'


def sex_str_to_int(
        s: Literal['m', 'f']
) -> int:
    """Get an integer value corresponding to gender letter in microcensus"""
    # maybe rather make a dictionary constant
    if s == 'm':
        return 0
    elif s == 'f':
        return 1
    raise RuntimeError(f'"{s}" is an unrecognized sex/gender')


def str_to_seconds(
        s: str
) -> int:
    """Translate HH:MM:SS string to integer seconds (floor of total seconds)"""
    return int(pd.to_timedelta(s).total_seconds())

# percentiles for deriving quantiles
QUANTILES = [0, 0.1, 0.25, 0.5, 0.75, 0.99, 1]
# string for pt interaction
PT_ACT = 'pt interaction'
# rename mapping for `matsim` values to correspond with `microcensus`
MATSIM_TO_MICROCENSUS: Dict[str, str] = {
    'age': 'age',
    'sex': 'sex',
    'employed': 'employed',
    'carAvail': 'car_availability',
    'hasLicense': 'driving_license',
    'isCarPassenger': 'is_car_passenger',
    'ptHasGA': 'subscriptions_ga',
    'ptHasHalbtax': 'subscriptions_halbtax',
    'ptHasStrecke': 'subscriptions_strecke',
    'ptHasVerbund': 'subscriptions_verbund',
    'spRegion': 'sp_region',
    'mode': 'mode',
    'end_time': 'departure_time',
    'dep_time': 'departure_time',  # for leg
    'start_time': 'arrival_time',
    'x': 'origin_x',
    'y': 'origin_y',
    # '': 'destination_x',
    # '': 'destination_y',
    # '': 'activity_duration',
    # '': 'crowfly_distance',
    'distance': 'network_distance',
    'type': 'purpose'
}


# translate `matsim` data types to `microcensus` ones
MICROCENSUS_DTYPES: Dict[str, Callable] = {
    'person_id': int,
    'age': int,
    'sex': sex_str_to_int,
    'driving_license': str_to_bool,
    'car_availability': str_to_bool,
    'employed': str_to_bool,
    'subscriptions_ga': str_to_bool,
    'subscriptions_halbtax': str_to_bool,
    'subscriptions_verbund': str_to_bool,
    'subscriptions_strecke': str_to_bool,
    'is_car_passenger': str_to_bool,
    'sp_region': int,
    'trip_id': int,
    'departure_time': str_to_seconds,
    'arrival_time': str_to_seconds,
    'mode': str,
    'type': str,
    'purpose': str,
    'distance': float,
    'destination_x': float,
    'destination_y': float,
    'origin_x': float,
    'origin_y': float,
    'activity_duration': float,
    'crowfly_distance': float,
    'parking_cost': float,
    'network_distance': float,
    'activity_duration': int
}

# those types are not in `microcensus`, but are needed for intermediate
# calculations to get `microcensus` values, particulartly `departure_time`
MATSIM_DTYPES = {
    'trav_time': str_to_seconds,
    'type': str  # activity type
}


def load_pickle(
        path: Union[str, Path]
) -> Any:
    """
    Returns whatever is saved in the binary pickle.

    Might fail on importing dependencies if not installed to the path known
    to the interpreter in use.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the pickled variable.

    Returns
    -------
    Any
        Whatever was unpickled

    """
    with open(path, mode='rb') as fp:
        unpickled = pickle.load(fp)
    return unpickled


def load_microcensus(
        eqasim_out_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load microcensus related .p files from directory with eqasim outputs.

    Parameters
    ----------
    eqasim_out_dir : Union[str, Path]
        Directory with eqasim outputs (.cache folders and .p files).

    Returns
    -------
    Dict[str, Any]
        Dictionary with datasets names as keys (data.microcensus.*).

    """
    microcensus_paths = list(
        Path(eqasim_out_dir).glob('data.microcensus.*.p')
    )
    microcensus_data = {}
    for mcpath in microcensus_paths:
        lastname = mcpath.stem.split('__')[0].split('.')[-1]
        unpickled = load_pickle(mcpath)
        microcensus_data[lastname] = unpickled
    return microcensus_data


def extract_route_attrs(
        leg: xml.etree.ElementTree.Element,
        leg_nr: int = 0,
        omit_veh_id: bool = True,
        drop_decimal_distance: bool = True,
        ignore_outsiders: bool = True,
        microcensus_style: bool = True,
        keep_keys: Optional[List[str]] = None
) -> Optional[Dict[str, Union[str, int, bool, float]]]:
    route_attrs = {}

    leg_dict = copy.deepcopy(leg.attrib)
    route_dict = copy.deepcopy(leg.find('route').attrib)
    if 'type' in route_dict:  # !!!
        del route_dict['type']

    if ignore_outsiders and leg_dict['mode'] == 'outside':  # mode
        return None  # differs from just an empty dictionary

    leg_dict.update(route_dict)
    for rkey, rvalue in leg_dict.items():
        if omit_veh_id and rkey == 'vehicleRefId':
            continue
        if drop_decimal_distance and rkey == 'distance':
            rvalue = rvalue.split('.')[0]
        if microcensus_style:
            # maybe a bit too complicated...
            if rkey in MATSIM_TO_MICROCENSUS:
                # rename the key if is in the mapping
                rkey = MATSIM_TO_MICROCENSUS[rkey]
                if rkey in MICROCENSUS_DTYPES:
                    # if renamed, convert to correct data type
                    route_attrs[rkey] = MICROCENSUS_DTYPES[rkey](rvalue)
            elif keep_keys is not None and rkey in keep_keys:
                # if not in the mapping, we can still include keys if needed
                if rkey in MATSIM_DTYPES:
                    # and possibly apply required data types
                    rvalue = MATSIM_DTYPES[rkey](rvalue)
                route_attrs[rkey] = rvalue
        else:
            route_attrs[f'{rkey}{leg_nr}'] = rvalue

    return route_attrs


def extract_act_attrs(
        act: bool,
        act_nr: int = 0,
        omit_veh_id: bool = True,
        drop_decimal_distance: bool = True,
        ignore_outsiders: bool = True,
        microcensus_style: bool = True,
        keep_keys: Optional[List[str]] = None
) -> Dict[str, Union[str, int, float, bool]]:
    act_attrs = {}

    for akey, avalue in act.attrib.items():
        if ignore_outsiders and akey == 'type' and avalue == 'outside':
            return
        if microcensus_style:
            if akey in MATSIM_TO_MICROCENSUS:
                akey = MATSIM_TO_MICROCENSUS[akey]
                if akey in MICROCENSUS_DTYPES:
                    act_attrs[akey] = MICROCENSUS_DTYPES[akey](avalue)
            elif keep_keys is not None and akey in keep_keys:
                if akey in MATSIM_DTYPES:
                    avalue = MATSIM_DTYPES[akey](avalue)
                act_attrs[akey] = avalue
        else:
            act_attrs[f'{akey}{act_nr}'] = avalue
    return act_attrs


def calculate_distance(
        dict1: Dict[str, Any],
        dict2: Dict[str, Any],
        dict1_origin_prefix: str = 'origin_',
        dict2_destination_prefix: str = 'origin_',
        drop_decimal_distance: bool = True
) -> Union[float, int]:
    # current origin
    ox = dict1[f'{dict1_origin_prefix}x']
    oy = dict1[f'{dict1_origin_prefix}y']
    orig = ox, oy
    # next origin, current destination
    dx = dict2[f'{dict2_destination_prefix}x']
    dy = dict2[f'{dict2_destination_prefix}y']
    dest = dx, dy
    # beeline distance, Pythagorean theorem
    beeline_flt = math.dist(orig, dest)
    if drop_decimal_distance:
        beeline = int(beeline_flt)
    else:
        beeline = beeline_flt
    return beeline


def get_person_attributes(
        person: xml.etree.ElementTree.Element,
        ignore_freight: bool = True
) -> Dict[str, Union[str, int, float, bool]]:
    person_data = {}

    for attr in person.find('attributes').findall('attribute'):
        for pkey, pvalue in attr.items():
            if (ignore_freight and
                    attr.attrib['name'] == 'isFreight' and
                        attr.text == 'true'):
                return None
            attr_name = attr.attrib['name']
            if attr_name not in MATSIM_TO_MICROCENSUS:
                continue
            attr_name = MATSIM_TO_MICROCENSUS[attr_name]
            if attr_name in MICROCENSUS_DTYPES:
                attr_value = MICROCENSUS_DTYPES[attr_name](attr.text)
            else:
                attr_value = attr.text
            person_data[attr_name] = attr_value
    return person_data


def filter_non_pt_acts(
        acts_attrs: List[Dict[str, Union[str, int, float]]],
        legs_attrs: List[Dict[str, Union[str, int, float]]]
) -> List[List[int]]:
    pt_related_legs_groups = []
    non_pt_acts = []
    last_act_pt = False
    has_pt = False
    for i, act_attrs in enumerate(acts_attrs):
        if act_attrs['purpose'] == PT_ACT:
            if i == 0:
                raise NotImplementedError(
                    f"{PT_ACT} activity is not supported to be first in an "
                    "agent's plan"
                )
            if not last_act_pt:
                pt_related_legs_groups.append([])
            pt_related_legs_groups[-1].append(i - 1)
            last_act_pt = True
            has_pt = True
        else:
            if last_act_pt:
                pt_related_legs_groups[-1].append(i - 1)
            elif pt_related_legs_groups:
                pt_related_legs_groups.append(i - 1)
            non_pt_acts.append(act_attrs)
            last_act_pt = False

    if not has_pt:
        return acts_attrs, legs_attrs

    non_pt_legs = []
    for leg_group in pt_related_legs_groups:
        if isinstance(leg_group, int):
            non_pt_legs.append(legs_attrs[leg_group])
        else:
            group_legs_attrs = [
                item for i, item in enumerate(legs_attrs) if i in leg_group
            ]
            sum_net_dist = sum(
                item['network_distance'] for item in
                group_legs_attrs
            )
            sum_trav_time = sum(
                item['trav_time'] for item in
                group_legs_attrs
            )
            group_leg_attrs = copy.deepcopy(legs_attrs[leg_group[0]])
            group_leg_attrs['network_distance'] = sum_net_dist
            group_leg_attrs['trav_time'] = sum_trav_time
            group_leg_attrs['mode'] = 'pt'
            non_pt_legs.append(group_leg_attrs)

    return non_pt_acts, non_pt_legs


def get_person_data_as_microcensus(
        person: xml.etree.ElementTree.Element,
        plan: xml.etree.ElementTree.Element,
        ignore_freight: bool = True,
        ignore_single_act: bool = False,
        ignore_outsiders: bool = True,
        omit_veh_id: bool = True,
        drop_decimal_distance: bool = True
) -> Optional[List[Dict[str, Union[str, bool, int]]]]:

    person_attributes = get_person_attributes(
        person=person,
        ignore_freight=ignore_freight
    )
    if not person_attributes:
        return None
    person_data = dict(
        person_id=int(person.attrib['id']),
        **person_attributes
    )

    acts = plan.findall('activity')
    legs = plan.findall('leg')

    if ignore_single_act and len(acts) == 0:
        return None
    if len(legs) != len(acts) - 1:
        raise RuntimeError(
            'Number of legs does not correspond to number of activities'
        )

    acts_attrs = [
        extract_act_attrs(
            act=act,
            omit_veh_id=omit_veh_id,
            drop_decimal_distance=drop_decimal_distance,
            ignore_outsiders=ignore_outsiders,
            microcensus_style=True
        ) for act in acts
    ]
    legs_attrs = [
        extract_route_attrs(
            leg=leg,
            omit_veh_id=omit_veh_id,
            drop_decimal_distance=drop_decimal_distance,
            ignore_outsiders=ignore_outsiders,
            microcensus_style=True,
            # keep this to calculate next start time
            keep_keys=['trav_time']  # !!! do we need it?
        ) for leg in legs
    ]
    if None in acts_attrs:
        return None
    if None in legs_attrs:
        return None

    acts_attrs, legs_attrs = filter_non_pt_acts(
        acts_attrs=acts_attrs,
        legs_attrs=legs_attrs
    )

    list_person_data = []
    stage_num = 1
    for i, leg_attrs in enumerate(legs_attrs):
        curr_act = acts_attrs[i]
        next_act = acts_attrs[i + 1]

        stage_attrs = copy.deepcopy(leg_attrs)
        stage_attrs['origin_x'] = curr_act['origin_x']
        stage_attrs['origin_y'] = curr_act['origin_y']
        stage_attrs['person_id'] = int(person.attrib['id'])
        # next origin becomes current destination
        stage_attrs['destination_x'] = next_act['origin_x']
        stage_attrs['destination_y'] = next_act['origin_y']
        distance = calculate_distance(
            dict1=curr_act,
            dict2=next_act,
            drop_decimal_distance=drop_decimal_distance
        )
        stage_attrs['crowfly_distance'] = distance
        stage_attrs['arrival_time'] = next_act['arrival_time']
        stage_attrs['mode'] = leg_attrs['mode']

        if i < len(acts_attrs) - 2:
            stage_attrs['activity_duration'] = acts_attrs[i + 1]['departure_time'] - next_act['arrival_time']
        else:
            stage_attrs['activity_duration'] = None
        stage_attrs['trip_id'] = stage_num
        stage_attrs['activity'] = curr_act['purpose']
        stage_attrs['purpose'] = next_act['purpose']
        stage_attrs['person_id'] = int(person.attrib['id'])
        list_person_data.append(stage_attrs)
        stage_num += 1
    return list_person_data


# def check_person_data(
#         list_person_data: List[Dict[str, Any]]
# ):
#     for num, pd_stage in enumerate(list_person_data[:-1]):
#         next_pd_stage = list_person_data[num + 1]
#         trtime = pd_stage['arrival_time'] - pd_stage['departure_time']
#         assert pd_stage['trav_time'] == trtime
        

def get_person_data(
        person: xml.etree.ElementTree.Element,
        plan: xml.etree.ElementTree.Element,
        microcensus_style: bool = True,
        ignore_freight: bool = True,
        ignore_single_act: bool = False,
        ignore_outsiders: bool = True,
        omit_veh_id: bool = True,
        drop_decimal_distance: bool = True
) -> Optional[Dict[str, Union[str, bool, int]]]:
    """
    Extract data from person and his/her plan as a dictionary of strings.

    If some of ignoring conditions trigger, None is returned. Omitting
    conditions skip when triggered, but don't return None. Plan's keys that
    repeat get inclemental integer suffixes. In case microcensus styling,
    those integers go to `trip_id` to each person.

    Parameters
    ----------
    person : xml.etree.ElementTree.Element
        A person's element from ``matsim.plan_reader()``.
    plan : xml.etree.ElementTree.Element
        A person's plan element from ``matsim.plan_reader()``.
    microcensus_style : bool, optional
        Whether to make the output row to use the same layout as microcensus.
        The default is True.
    ignore_freight : bool, optional
        Return None if ``isFreight`` tag is present and is ``true``. The
        default is True.
    ignore_single_act : bool, optional
        Return None if ``activity`` tag is present only once in plan or if no
        activities at all. The default is False.
    ignore_outsiders : bool, optional
        Return None if ``type`` of activity or ``mode`` of leg is ``outside``.
        The default is True.
    omit_veh_id : bool, optional
        Don't include ``vehicleRefId``s in the output. The default is True.
    drop_decimal_distance: bool, optional
        Don't include decimal digits within ``distance`` key of leg's route.
        The default is True.

    Returns
    -------
    Optional[Dict[str, str]]

    """
    id_attr = 'person_id' if microcensus_style else 'id'
    person_data = {
        id_attr: person.attrib['id']
    }
    list_person_data = []  # only for microcensus style
    for attr in person.find('attributes').findall('attribute'):
        for pkey, pvalue in attr.items():
            if (ignore_freight and
                    attr.attrib['name'] == 'isFreight' and
                        attr.text == 'true'):
                return
            attr_name = attr.attrib['name']
            if microcensus_style and attr_name in MATSIM_TO_MICROCENSUS:
                attr_name = MATSIM_TO_MICROCENSUS[attr_name]
            elif microcensus_style and attr_name not in MATSIM_TO_MICROCENSUS:
                continue
            person_data[attr_name] = attr.text

    acts = plan.findall('activity')
    legs = plan.findall('leg')

    if ignore_single_act and len(acts) == 0:
        return None
    if len(legs) != len(acts) - 1:
        raise RuntimeError(
            'Number of legs does not correspond to number of activities'
        )

    # to make zip possible and preserve the last activity
    legs.append(None)  

    for i, (act, leg) in enumerate(zip(acts, legs)):
        act_attrs = extract_act_attrs(
            act=act,
            act_nr=i,
            omit_veh_id=omit_veh_id,
            drop_decimal_distance=drop_decimal_distance,
            ignore_outsiders=ignore_outsiders,
            microcensus_style=microcensus_style
        )
        if act_attrs is None:
            # if encounered outside and it's not set as possible
            return None

        if microcensus_style and i > 0:
            # previous origin

            ox = list_person_data[-1]['origin_x']
            oy = list_person_data[-1]['origin_y']
            orig = ox, oy
            # current origin, previous destination
            dx = act_attrs['origin_x']
            dy = act_attrs['origin_y']
            dest = dx, dy
            # beeline distance, Pythagorean theorem
            beeline_flt = math.dist(orig, dest)
            if drop_decimal_distance:
                beeline = int(beeline_flt)
            else:
                beeline = beeline_flt
            act_attrs['crowfly_distance'] = beeline

        # !!! TODO change conditioning, this is horrible...

        if leg is not None:
            # not last iteration
            route_attrs = extract_route_attrs(
                leg=leg,
                leg_nr=i,
                omit_veh_id=omit_veh_id,
                drop_decimal_distance=drop_decimal_distance,
                ignore_outsiders=ignore_outsiders,
                microcensus_style=microcensus_style,
                # keep this to calculate next start time
                keep_keys=['trav_time']
            )
            if route_attrs is None:
                return None

            if microcensus_style:
                new_person_data = copy.deepcopy(person_data)
                new_person_data.update(route_attrs)
                new_person_data.update(act_attrs)
                new_person_data['arrival_time'] = (
                    new_person_data['departure_time'] +
                    new_person_data['trav_time']
                )
                del new_person_data['trav_time']
                list_person_data.append(new_person_data)
            else:
                person_data.update(route_attrs)
                person_data.update(act_attrs)

    if microcensus_style:
        return list_person_data
    return person_data


def get_population_data(
        population_path: Union[str, Path],
        microcensus_style: bool = True
) -> pd.DataFrame:
    pre_rows = []
    reader = matsim.plan_reader(population_path, selected_plans_only=True)

    for person, plan in reader:
        person_data = get_person_data_as_microcensus(
            person=person,
            plan=plan,
            ignore_freight=True,
            ignore_outsiders=True,
            ignore_single_act=False,
            omit_veh_id=True,
            drop_decimal_distance=True
        )
        if person_data is not None:
            pre_rows.append(
                pd.DataFrame(person_data)
            )

    if microcensus_style:
        population_data = pd.concat(pre_rows).reset_index(drop=True)
    else:
        population_data = pd.DataFrame(pre_rows)
    return population_data


def cut_microcensus(
        microcensus_data: Dict[str, Any],
        shp_path: Union[str, Path],
        cut_trips: bool = True,
        cut_work_commute: bool = True,
        cut_education_commute: bool = True,
        cut_households: bool = True,
        cut_persons: bool = True
) -> Dict[str, Any]:
    extent = gpd.read_file(shp_path)
    extent_poly = extent.unary_union
    cut_microcensus_data = {}

    if cut_trips:
        trips = microcensus_data['trips'][0].reset_index(drop=True)

        origins = gpd.GeoSeries(
            gpd.points_from_xy(x=trips['origin_x'], y=trips['origin_y'])
        )
        origins.set_crs(crs=extent.crs)
        dests = gpd.GeoSeries(
            gpd.points_from_xy(x=trips['destination_x'], y=trips['destination_y'])
        )
        dests.set_crs(crs=extent.crs)

        trips['origin'] = origins
        trips['destination'] = dests

        origins_within = origins.within(extent_poly)
        dests_within = dests.within(extent_poly)
    
        trips.loc[origins_within & dests_within, 'within'] = True
        trips.loc[trips['within'].isna(), 'within'] = False
    
        persons_to_remove = set()
        for person_id, person_trips in trips.groupby('person_id'):
            if not person_trips['within'].all():
                persons_to_remove.add(person_id)

        cut_trips = trips[~trips['person_id'].isin(list(persons_to_remove))].copy()
        # !!! TODO: figure out trip_numbers
        cut_trips_numbers = microcensus_data['trips'][1]
        cut_microcensus_data['trips'] = (cut_trips, cut_trips_numbers)
    else:
        cut_microcensus_data['trips'] = microcensus_data['trips']

    # if cut_persons:
    #     persons = microcensus_data['persons']

    # if cut_work_commute:
    #     work_commute = microcensus_data['commute']['work']
    #     commute_xy = gpd.GeoSeries(
    #         gpd.points_from_xy(x=trips['destination_x'], y=trips['destination_y'])
    #     )
    #     commute_xy.set_crs(crs=extent.crs)
    #     work_commute['commute'] = commute_xy

    return cut_microcensus_data


def microcensus_to_population(
        microcensus_data: Dict[str, Any]
) -> pd.DataFrame:
    microcensus_data['trips'][0].columns
    # TODO


def get_value_stats_by_mode(
        dfs: List[pd.DataFrame],
        value_name: str,
        value_thresh_plot: Optional[Union[float, int]] = None
) -> Tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """
    Extract stats from a list of microcensus style dataframes by some column.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        DataFrames in microcensus style.
    value_name : str
        Name of the column fro stats to be based on.
    value_thresh_plot : Optional[Union[float, int]], optional
        Under what value should the results in plot be. The default is None.

    Returns
    -------
    Tuple[pd.DataFrame, matplotlib.figure.Figure]
        The data as DataFrame and plot as matplotlib axes.

    """
    stats_dicts = []
    for df in dfs:
        for mode, modedata in df.groupby('mode'):
            if value_name not in modedata.columns:
                continue
            stats_dict = {}
            qs = modedata[value_name].quantile(QUANTILES).to_dict()
            mean = modedata[value_name].mean()
            std = modedata[value_name].std()
            stats_dict['source'] = modedata['source'].iloc[0]
            stats_dict['mode'] = mode
            stats_dict['stdev'] = std
            stats_dict['mean'] = mean
            for q, val in qs.items():
                stats_dict[f'q{int(q * 100)}'] = val
            stats_dicts.append(stats_dict)
    stats_data = pd.DataFrame(stats_dicts)

    combined_dfs = pd.concat(dfs).reset_index(drop=True)
    if value_thresh_plot is not None:
        combined_dfs = combined_dfs[
                combined_dfs[value_name] <  value_thresh_plot
        ]
    ax = combined_dfs.boxplot(
        column=value_name,
        by=['source', 'mode'],
        rot=90
    )
    return stats_data, ax



def get_modal_split_by_distance(
        dfs: List[pd.DataFrame],
        distance_step: Union[int, float] = 1000
) -> Tuple[pd.DataFrame, matplotlib.figure.Figure]:
    combined_dfs = pd.concat(dfs).reset_index(drop=True)
    combined_dfs = combined_dfs[~combined_dfs['crowfly_distance'].isna()]
    max_dist = np.ceil(
        combined_dfs['crowfly_distance'].max() / distance_step
    ) * distance_step    
    dist_wins = [
        i * distance_step for i in range(int(max_dist / distance_step) + 1)
    ]
    cats = pd.cut(combined_dfs['crowfly_distance'], dist_wins)
    combined_dfs['category'] = cats
    modes = combined_dfs['mode'].unique()

    rows = []
    for (cat, src), df_mcs in combined_dfs.groupby(['category', 'source']):
        if pd.isna(cat):
            continue
        modes_cnt = {
            f'{m}_abs': len(df_mcs[df_mcs['mode'] == m]) for m in modes
        }
        modes_sum = sum(modes_cnt.values())
        modes_rel = {
            f'{m}_rel': modes_cnt[f'{m}_abs'] / modes_sum for m in modes
        }
        row = {
            'source': src,
            'less_than_m': cat.right,
            'count': modes_sum
        }
        row.update(modes_rel)
        rows.append(row)
    distance_ms = pd.DataFrame(rows)

    cmap = matplotlib.colormaps['gist_rainbow'](np.linspace(0, 1, len(modes)))
    modes_colors = dict(zip(modes, cmap))
    
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    sources = list(combined_dfs['source'].unique())
    sources_linestyle = dict(zip(sources, linestyles))
    
    if len(linestyles) < len(sources):
        raise RuntimeError('There are too many sources to be plotted')

    fig, ax = plt.subplots(layout='tight')
    for src, src_df in distance_ms.groupby('source'):
        linetype = sources_linestyle[src]
        for mode in modes:
            x = src_df['less_than_m'].tolist()
            y = src_df[f'{mode}_rel'].tolist()
            color = modes_colors[mode]
            ax.plot(x, y, color=color, linestyle=linetype, label=f'{mode} ({src})')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(1.05, 1.25),
        ncol=len(modes),
        fancybox=True,
        shadow=True,
    )
    return distance_ms, ax


def get_modal_split(
        dfs: List[pd.DataFrame]
) -> Tuple[pd.DataFrame, matplotlib.figure.Figure]:
    combined_dfs = pd.concat(dfs).reset_index(drop=True)
    ms_abs = combined_dfs.groupby('source')['mode'].value_counts()
    ms_rel = ms_abs / ms_abs.groupby('source').sum()

    sources = list(combined_dfs['source'].unique())
    modes = list(combined_dfs['mode'].unique())
    source_modes_rel = defaultdict(list)

    for src in sources:
        for mode in modes:
            source_modes_rel[src].append(ms_rel[(src, mode)])

    x = np.arange(len(modes))  # the label locations
    width = 1 / (len(sources) + 1) # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    for pos, (attribute, measurement) in enumerate(source_modes_rel.items()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset, np.round(measurement, 2), width, label=attribute
        )
        ax.bar_label(rects, padding=3, size=8)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Share')
    ax.set_title('Modal split of all legs')
    ax.set_xticks(x + (width * (len(sources) - 1) / 2), modes)
    ax.legend(loc='upper center', ncols=len(sources))
    ax.set_ylim(0, ms_rel.max() + 0.1)

    ms_rel_df = ms_rel.to_frame().rename({'mode': 'share'}, axis=1).reset_index()
    return ms_rel_df, ax


def main(
        microcensus_dir: Union[str, Path],
        population_path: Union[str, Path]
):
    microcensus_dir = (
         '/home/leonefamily/csfm/runs/switzerland/'
    )
    # population_path = (
    #     '/home/leonefamily/csfm/runs/switzerland/'
    #     'matsim.simulation.run__323656762bf1d320c148278807d409a6.cache/'
    #     'simulation_output_cut/zurich_population.xml.gz'
    # )
    population_path = "/home/leonefamily/csfm/vdf_vs_normal/normal/output_plans.xml.gz"
    vdf_population_path = "/home/leonefamily/csfm/vdf_vs_normal/vdf/output_plans.xml.gz"
    vdf_reroute_population_path = "/home/leonefamily/csfm/vdf_vs_normal/vdf_reroute/output_persons.csv.gz"
    shp_path = (
        '/home/leonefamily/csfm/runs/switzerland/'
        'matsim.runtime.eqasim__03a9d891a6bec1c43e9e04342e518596.cache/'
        'eqasim-java/gis/zurich_city.shp'
    )

    population_data = get_population_data(population_path)
    vdf_population_data = get_population_data(vdf_population_path)
    vdf_reroute_population_data = get_population_data(vdf_reroute_population_path)
    microcensus_data = load_microcensus(eqasim_out_dir=microcensus_dir)

    cut_microcensus_data = cut_microcensus(
        microcensus_data=microcensus_data,
        shp_path=shp_path,
        cut_trips=True
    )

    micro_trips = cut_microcensus_data['trips'][0].copy()
    micro_trips['trav_time'] = (
        micro_trips['arrival_time'] - micro_trips['departure_time']
    )
    micro_trips['source'] = 'MZ'
    population_data['source'] = 'sim'
    vdf_population_data['source'] = 'vdf'
    vdf_reroute_population_data['source'] = 'vdf'

    dists_modes_data, dists_modes_plot = get_value_stats_by_mode(
        dfs=[population_data, vdf_population_data, vdf_reroute_population_data, micro_trips],
        value_name='crowfly_distance',
        value_thresh_plot=25000
    )
    trtime_modes_data, trtime_modes_plot = get_value_stats_by_mode(
        dfs=[population_data, vdf_population_data, vdf_reroute_population_data, micro_trips],
        value_name='trav_time',
        value_thresh_plot=15000
    )
    modal_split_distance_data, modal_split_distance_plot = get_modal_split_by_distance(
        dfs=[population_data, vdf_population_data, vdf_reroute_population_data, micro_trips],
        distance_step=1000
    )
    modal_split_data, modal_split_plot = get_modal_split(
        dfs=[population_data, vdf_population_data, vdf_reroute_population_data, micro_trips],
    )
